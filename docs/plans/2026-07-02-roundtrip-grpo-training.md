---
type: plan
status: current
created: 2026-07-02
updated: 2026-07-02
tags: [d2, grpo, round-trip-reward, rl, substitutor, co-design, plan]
companion: [2026-07-02-surrogate-grpo-training.md, 2026-07-02-codesign-next-stage.md, 2026-07-02-d1-prototype-implementation.md, ../research/adverserial-RL.md]
---

# Round-trip GRPO training of the substitutor (D2 · Way 1)

**Scope update (2026-07-02, end of design session):** the active v0 of Way 1 trains on a
**model-free local surrogate reward** instead of this plan's remote round trip — see
[`2026-07-02-surrogate-grpo-training.md`](2026-07-02-surrogate-grpo-training.md), which imports
this plan's decided forks (staged policy, hand-rolled GRPO, MTI head, scalar α) and keeps the true
round trip as evaluation. This document remains the full round-trip-reward design for future
revisit; the surrogate plan's "Kill / upgrade criteria" section defines when to return here
(surrogate blind spots — phrasing-induced remote failures — dominating the residual, or
multi-task/multi-model training justifying the rollout cost).

D2 ([`2026-07-02-codesign-next-stage.md`](2026-07-02-codesign-next-stage.md)) splits into two
variants sharing one objective, the round-trip reward:

- **Way 1 (this plan):** one-time **RL training** (GRPO) of the substitutor cascade's learnable
  components against the round-trip reward. Pays training cost once; at deployment the substitutor
  runs in a single local forward pass, no extra remote calls.
- **Way 2 (separate session):** **training-free per-document search** over substitution choices,
  scoring candidates with the same reward — n extra round trips per document. Serves as Way 1's
  **go/no-go probe** (if direct search with the true reward can't shift the Pareto frontier vs the
  τ-walk, a policy trained on that reward won't either) and as an SFT/DPO data generator.
  Handoff: [`2026-07-02-roundtrip-search-probe.md`](../handoffs/2026-07-02-roundtrip-search-probe.md).

**Gate (from the parent plan, unchanged):** GRPO compute is spent only after Way 2 shows a frontier
shift at matched realized privacy.

## Definitions

- **Round-trip reward:** `r = α·(1 − A(doc_p)) + (1−α)·U(out_final)` where `A` is a local attack
  head's re-identification success on `doc_p` and `U` the utility of `out_final` (vs `out_ref` /
  gold answers). α is D2's privacy knob: fixed within a run, swept across runs for the Pareto curve.
- **GRPO (Group Relative Policy Optimization):** policy-gradient RL without a value network —
  sample a *group* of G outputs per input, use within-group reward standardization as the advantage.
- **Policy:** the trainable substitutor components (below). Everything else — detector, coref,
  lattices, extractor, attack head — is frozen.
- τ-walk, R, lattice, MTI probe, `out_ref`: see the
  [D1 plan's definitions](2026-07-02-d1-prototype-implementation.md#definitions).

## Mechanism — step by step

### Phase 0 — setup (once)

1. **Fix what's trainable.** The policy is *not* one big LLM; it's the cascade's two learnable parts:
   - the **candidate-selection ranker** — the model that picks a lattice level per span (this
     *replaces* the τ-walk heuristic). A `keep` level (no generalization) is in its action set, so it
     also **absorbs detector over-detection**: a recall-biased detector may surface non-PII spans
     (e.g. knowledgator's MISC firing on generic nouns —
     `docs/research/learned-PII-detection.md` §5.1d), and the ranker learns to keep them at zero
     privacy cost. This is why the detector is tuned for recall, not precision — over-detection is
     the ranker's job, and more surfaced spans mean more signal to learn from.
   - the **residual infiller** (flan-t5-class enc-dec, 250–780M) — generates replacement text for
     spans that have no lattice entry.
   Everything else is **frozen**: detector, coref, lattices, the D1 extractor, the attack head.
2. **SFT initialization** (supervised fine-tuning): train both components to imitate D1's traces
   (the τ-walk's choices and the teacher's infills), so the policy starts as "D1 cascade" rather
   than random. Way 2's searched solutions, if available, are strictly better SFT targets
   (behavior-clone the *searched* choices, not the heuristic's).
3. **Fix the reward pieces** (decisions in "Reward design" below): attack head `A` = MTI probe
   (correlation-gated upgrade path), utility metric `U` = cheapest proxy that tracks the headline
   utility on D1 tuples, weight `α` fixed per run. Cache all proxy calls.

### Phase 1 — GRPO loop (per document, per training step)

1. **Sample a group.** For document `d`, sample G (e.g. 8) *different* substitution outputs from
   the current policy — different lattice-level picks and infills → candidates
   `doc_p⁽¹⁾…doc_p⁽ᴳ⁾`, each with its record `R⁽ⁱ⁾`. (Sampling temperature is what makes the
   group diverse.)
2. **Roll out each candidate through the real round trip:**
   `out_p⁽ⁱ⁾ = RemoteLLM(doc_p⁽ⁱ⁾, task)` → `out_final⁽ⁱ⁾ = Extractor(out_p⁽ⁱ⁾, R⁽ⁱ⁾, doc_orig)`
   (extractor frozen).
3. **Score each:** `r⁽ⁱ⁾ = α·(1 − A(doc_p⁽ⁱ⁾)) + (1−α)·U(out_final⁽ⁱ⁾)`.
4. **Group-relative advantage** (GRPO's trick — no value network):
   `adv⁽ⁱ⁾ = (r⁽ⁱ⁾ − mean(r)) / std(r)`, computed *within the group of G siblings for the same
   document*. "Better than your siblings" is the learning signal, which cancels out per-document
   difficulty.
5. **Policy-gradient update:** increase the log-probability of the choices (level picks, infill
   tokens) inside high-advantage candidates, decrease for low-advantage ones (PPO-style clipped
   ratio). Update ranker + infiller weights.
6. Repeat over the corpus for a few epochs; α stays fixed within a run — sweeping α across runs
   traces the Pareto curve.

### The one subtlety

The reward's adversary `A` is a **cheap local stand-in**, not the real evaluation attacker. If
training and evaluation used the same attacker, the policy would overfit that attacker's blind
spots and the privacy number would be circular (RUPTA's flaw). So: train against the local head,
**evaluate against a held-out frontier-LLM attacker from a different model family** — the gap
between them is itself a finding. Second gotcha (flagged in the D1 plan): `trl`'s GRPOTrainer only
supports causal LMs, so the enc-dec infiller can't drop in as-is — resolved by the 2026-07-02
tooling survey: hand-roll the loop (see "RL tooling — resolved" below).

### Worked example

Doc `d`: "Kari, a nurse from Oslo, asks…", task = summarize. G=4 rollouts at α=0.5:

| i | policy's choices | A(doc_p) | U(out_final) | r | adv |
|---|---|---|---|---|---|
| 1 | Oslo→"a Norwegian city", nurse kept | 0.20 | 0.90 | 0.85 | +1.1 |
| 2 | Oslo kept, nurse kept | 0.80 | 0.95 | 0.575 | −1.0 |
| 3 | Oslo→"a European city", nurse→"a professional" | 0.10 | 0.60 | 0.75 | +0.3 |
| 4 | Oslo→"a place", nurse→"a person" | 0.05 | 0.40 | 0.675 | −0.4 |

Update: push the ranker toward rollout 1's pattern (coarsen the location one notch, keep the
occupation the task needs) and away from 2 (leaky) and 4 (over-coarsened, utility died). After
many documents, the policy has internalized "which spans to coarsen how far, given the task" —
knowledge the τ-walk never had. Search (Way 2) rediscovers this per document with n paid round
trips; the trained policy *amortizes* it into its weights — one local pass, zero extra remote
calls, generalizing to unseen documents.

## RL tooling — resolved (2026-07-02 survey): hand-roll the GRPO loop

**No maintained framework supports encoder-decoder policies** for GRPO/PPO/RLOO. trl's
GRPOTrainer/DPOTrainer/PPOTrainer are causal-only (the legacy `is_encoder_decoder` paths were
removed); verl, OpenRLHF, and NeMo-Aligner all generate rollouts through vLLM/SGLang, and vLLM
cannot generate with T5/BART — enc-dec is outright broken on ROCm (all ROCm attention backends
are decoder-only: vLLM issues [#187](https://github.com/vllm-project/vllm/issues/187),
[#7366](https://github.com/vllm-project/vllm/issues/7366),
[#27442](https://github.com/vllm-project/vllm/issues/27442)) — so any vLLM-based stack is blocked
on this box regardless. The two frameworks that ever supported T5 are dead: TRLx (last push
2024-01, CarperAI defunct) and RL4LMs (frozen 2023-03, pinned old transformers), and both are
PPO (need a value head), not GRPO; resurrecting them = dependency surgery on a stack whose torch
must not be touched (ROCm rule), to get the wrong algorithm.

**Decision: hand-rolled GRPO on `AutoModelForSeq2SeqLM`, ~150–250 lines.** Smaller than for a
causal model: `generate(num_return_sequences=G, do_sample=True)` natively produces the rollout
group (no vLLM, no chat template), GRPO has no critic, and a 250M policy fits the iGPU. Loop =
sample G → score with round-trip reward → group-normalize advantages → one teacher-forced forward
pass for log-probs → clipped loss + KL-to-ref. Proof-of-shape: GRPO on seq2seq MT
([arXiv 2605.15976](https://arxiv.org/abs/2605.15976)); reference code to crib: TRLx's T5 PPO
branch (`modeling_ppo.py`). The switch-infiller-to-causal option is dropped — trl compatibility
buys nothing a 200-line loop doesn't, and costs re-distilling the teacher data.

## Joint two-component training — how (2026-07-02 survey)

Shared-reward policy gradients for a common-payoff team are **unbiased per component; the issue is
variance only**. Two case-specific mitigators: GRPO's group mean is already a strong baseline, and
the components have near-disjoint action surfaces (the infiller fires only on residue spans with
no lattice), so on many spans only one component's action varies. Ranked options from the survey:

1. **Shared group-relative advantage** *(default)*: one within-group advantage from the round-trip
   reward, applied to both the ranker's per-span action log-probs and the infiller's token
   log-probs, reusing the same G rollouts. Zero extra rollouts, no critic — the native
   two-component generalization of GRPO, as done by MMOA-RAG
   ([arXiv 2501.15228](https://arxiv.org/abs/2501.15228)) and MAGRPO
   ([arXiv 2508.04652](https://arxiv.org/abs/2508.04652)).
2. **Staged warm start / partner-at-greedy** *(scaffold + diagnostic)*: train the ranker first
   against the frozen greedy infiller (removes its sampling noise entirely), then unfreeze;
   partner-at-greedy runs isolate each component's variance (s3,
   [arXiv 2505.14146](https://arxiv.org/abs/2505.14146)). Also yields the selection-vs-infilling
   attribution ablation for free. Pure alternation alone can stall at a poor joint optimum —
   complement to 1, not a replacement.
3. **Escalation only if measured gradient variance blocks learning:** COMA counterfactual critic
   ([arXiv 1705.08926](https://arxiv.org/abs/1705.08926)) or selective difference rewards
   ([arXiv 2012.11258](https://arxiv.org/abs/2012.11258)) — both cost a critic or extra paid
   round trips; don't pay on spec. (Per the honesty rules: optimizer choice may use variance
   diagnostics; method *claims* only at matched realized privacy on `out_final`.)

## Policy scope — decided (2026-07-02 grilling): staged, ranker → joint

**Stage 1:** train the ranker alone against the frozen greedy infiller (partner-at-greedy,
s3-style). The ranker in isolation is a contextual bandit — per-span level choice, scalar reward,
plain-torch update, no generation sampling — with the cleanest possible credit assignment.
**Stage 2:** unfreeze the infiller; train both with the shared group-relative advantage
(MAGRPO-style, same G rollouts for both loss terms) via the hand-rolled seq2seq GRPO loop.
Stage 1 alone is already a reportable Pareto point; stage 2 − stage 1 is the measured value of
adapting the infiller, i.e. the selection-vs-infilling attribution comes free with the staging.

## Reward design — forks, decisions, and kept alternatives (2026-07-02 grilling)

Unchosen options below are documented deliberately: each is a candidate to revisit if the chosen
route underperforms.

### Fork 1 — local attack head `A`: decided (a), MTI + correlation-gated upgrade

`A` scores every rollout candidate's re-identifiability — tens of thousands of calls per training
run, so it must be cheap; the frontier-LLM attacker stays evaluation-only (cost + circularity).
The managed risk: the policy learns to fool `A` specifically — any blind spot of `A` becomes fake
privacy. Fidelity (does `A` rank candidates like the real attacker?) trades against per-candidate
cost in the hot loop.

- **(a) MTI probe + correlation-gated upgrade** *(decided)*: existing probe
  (`src/cloak/probe.py`, roberta-base, batches). Blind spot: token-level guess-back can't see
  document-level QI-combination attacks ("nurse + Oslo + age 34"). Upgrade rule, empirical: on
  D1's τ-sweep points, compare each head's ranking of operating points with the held-out eval
  attacker's; `A` = cheapest head that agrees.
- **(b) Document-level attribute-inference encoder** (DeBERTa-class on SynthPAI's 8 gold
  attributes): sees exactly what MTI can't; costs an up-front training project, and its quality
  becomes a hidden confound inside the reward. **Head survey (2026-07-02, agent research pass):
  no reusable off-the-shelf head exists.** All SynthPAI-native frameworks run *prompted* LLM
  attackers — [eth-sri/llm-anonymization](https://github.com/eth-sri/llm-anonymization) (GPT-4),
  AgentStealth (DeepSeek-V3 judge, no released weights), SEAL
  ([code](https://github.com/kykim0/SEAL) only, no attacker checkpoints; critique distilled into
  its 8B anonymizer, not a standalone attacker); HF hub offers at best single-attribute heads
  (gender ~0.4B, PAN-style age/gender) missing SynthPAI's discriminating attributes. So (b) =
  train from scratch — cheap ingredients (SynthPAI gold labels, multi-head encoder finetune,
  fits the iGPU) but a real subproject with its own validation burden.
- **(c) Prompted local-LLM attacker on llama-swap**: highest fidelity; most expensive reward term
  (LLM call per candidate × G), one-GPU contention with training, and role-hygiene overlap
  (E4B = teacher, Qwen = remote task model).

### Fork 2 — utility term `U`: decided (a), measure on D1 tuples

`U(out_final)` runs per candidate → needs a cheap proxy for the headline utility. Structural note:
AgentStealth's under-anonymization pathology (utility = `doc_p` similarity to `doc_orig`) cannot
occur here — `U` is computed on `out_final` after the full round trip; the residual risk is only
proxy-metric drift (e.g. lexical overlap rewarding verbatim copying).

- **(a) Decided:** on D1 tuples, correlate ROUGE-L / BERTScore / EM-F1 against the headline
  utility across τ points; adopt the cheapest metric that tracks, per task family (likely EM/F1
  for QA — gold-grounded, free; ROUGE-L or BERTScore for summarization). One analysis script.
- *(b, later)* fix BERTScore: semantically robust single choice; GPU pass per candidate
  (contention), possibly overkill.
- *(c, later)* fix ROUGE-L: free, CPU-only; purely lexical — brittle exactly where the round trip
  paraphrases.

### Fork 3 — reward form / privacy knob: decided (a), scalar α

The combination rule doubles as D2's privacy knob (the swept quantity for the Pareto curve, which
the honesty rules require to be the method's legitimate knob).

- **(a) Decided:** `r = α·(1−A) + (1−α)·U`, α fixed within a run, swept across runs (~3 values →
  3 Pareto points). Declared D2 knob in the parent plan's eval protocol, smooth dense reward
  (GRPO-friendly), AgentStealth-recipe compatible. Cost: one training run per operating point;
  realized-privacy placement discovered post-hoc.
- *(b, fallback — documented, not improvised mid-run)* RUPTA-style lexicographic: push attacker
  success below threshold T, then maximize utility subject to it. Aims directly at
  matched-privacy levels (T is the knob), but the thresholded reward is discontinuous —
  sparse/spiky gradients near the boundary are a known GRPO failure mode — and the phase schedule
  adds hyperparameters. Trigger to revisit: the α sweep's realized-privacy points come out
  clustered or degenerate.

## Prerequisites (all before any GRPO run)

- **D1 P4 complete:** τ-sweep Pareto + eval harness — the baseline the frontier shift is measured
  against; rung B (or rung A, if D1's kill criterion fired) as the frozen extractor.
- ~~SEAL registered + overlap assessed~~ **done 2026-07-02**:
  [wiki page](../../research-wiki/papers/kim2025_seal_adversarial_distillation.md)
  ([arXiv 2506.01420](https://arxiv.org/abs/2506.01420)). Verdict: no contribution collision
  (text-release, no round trip, no R/extractor, monolithic 8B); strong adjacency — their
  trajectories/recipe are candidate SFT/preference data, critique distillation is the buildable
  escalation attack head, and their 8B≈GPT-4 result is the comparison bar for `doc_p` privacy.
- **Way 2 probe result** — the gate.

## Wall-time estimate (2026-07-02; v0.1 scale = 60 SynthPAI docs × 2 tasks, G=8, ~5 epochs, 3 α)

Scheduling fact: the round-trip model (`Qwen3.6-35B-A3B`) is served by the **remote ts-llm-proxy —
its calls don't touch the iGPU**, so rollouts overlap with local training; only MTI, the
extractor, and policy updates share the GPU. **Assumption to calibrate at perf-gate time:
sustained proxy throughput ≈ 1 call/s** with `pmap` threading (unmeasured; 15-min calibration
run first).

- **Stage 1 (ranker bandit):** 60×2×8×5 = 4,800 reward evals per α, ×3 α = 14,400; bandit
  convergence means later epochs resample identical `doc_p` → `LLMClient` cache hits → ~6–9k
  unique calls ≈ **2–2.5 h proxy time total** (~45 min/α). Local side (rung-A rules, batched MTI,
  U, bandit update) negligible. **≈ half a day wall-clock.**
- **Stage 2 (joint GRPO):** infiller SFT warm start = D1 rung-B LoRA (2–4 h GPU, once; may
  already exist). Reward evals same formula but sampled infills defeat caching: ~4,800/α ≈
  1.3 h/α proxy (**~4 h** for 3 α) + ~1–3 h/α local GPU (flan-t5-base G-sample generation + LoRA
  backward), overlappable with proxy waits. **≈ 1–2 part-time days.**
- **Scaling:** 300 docs multiplies the proxy side ×5 (stage 1 ~1 day, stage 2 ~3–5 part-time
  days) — stay at 60 docs until a frontier shift is visible. Budget levers: α-count (one run per
  Pareto point) and epochs; if the U measurement picks BERTScore, each eval adds a small GPU
  forward on the shared iGPU.

## Risks (carried from the parent plan)

- **Rollout cost:** every sample is a real round trip (proxy + extractor + attack head);
  G× multiplies it. Batch and cache proxy calls (`LLMClient` cache); estimate wall-time against
  the perf gate before launch.
- **Reward hacking / R as leak channel:** the substitutor can collude with the extractor through R
  (stuff recoverable specifics into R while `doc_p` looks clean — privacy term unaffected, utility
  inflated). Audit R's information content as part of eval; leak-through check on `out_final`
  stays mandatory.
- **Small RL headroom:** AgentStealth measured +1.1 pt for RL over SFT. If Way 2's searched-SFT
  already captures the gain, GRPO may add nothing — that outcome is reportable per the honesty
  rules. **Kill criterion:** no frontier shift over D1 at matched realized privacy → D1's
  interface was the contribution; report and stop.

## Sources

Parent: [`2026-07-02-codesign-next-stage.md`](2026-07-02-codesign-next-stage.md) (D2, H2, sources
list); [`2026-07-02-d1-prototype-implementation.md`](2026-07-02-d1-prototype-implementation.md)
(roles, component stack, trl constraint). Method anchors:
[AgentStealth](../../research-wiki/papers/shao2025_agentstealth.md)
([arXiv 2506.22508](https://arxiv.org/abs/2506.22508)) — GRPO recipe + SFT-first;
[RUPTA](../../research-wiki/papers/yang2025_rupta.md)
([arXiv 2407.11770](https://arxiv.org/abs/2407.11770)) — evaluator/attacker circularity,
lexicographic schedule;
[SEAL](../../research-wiki/papers/kim2025_seal_adversarial_distillation.md)
([arXiv 2506.01420](https://arxiv.org/abs/2506.01420)) — closest neighbor, overlap assessed
2026-07-02, no collision.

Tooling/optimization surveys (2026-07-02, agent research passes): trl
[GRPO](https://huggingface.co/docs/trl/main/en/grpo_trainer)/[DPO](https://huggingface.co/docs/trl/main/en/dpo_trainer)/[PPO](https://huggingface.co/docs/trl/main/en/ppo_trainer)
docs; [TRLx](https://github.com/CarperAI/trlx); [RL4LMs](https://github.com/allenai/RL4LMs);
seq2seq GRPO ([arXiv 2605.15976](https://arxiv.org/abs/2605.15976)); MMOA-RAG
([arXiv 2501.15228](https://arxiv.org/abs/2501.15228)); MAGRPO
([arXiv 2508.04652](https://arxiv.org/abs/2508.04652)); s3
([arXiv 2505.14146](https://arxiv.org/abs/2505.14146)); COMA
([arXiv 1705.08926](https://arxiv.org/abs/1705.08926)); Dr.Reinforce
([arXiv 2012.11258](https://arxiv.org/abs/2012.11258)).
