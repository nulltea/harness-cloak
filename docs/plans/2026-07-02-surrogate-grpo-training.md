---
type: plan
status: current
created: 2026-07-02
updated: 2026-07-03
tags: [d2, grpo, surrogate-reward, qa-answerability, nli, rl, substitutor, task-eval, clinical, email, plan]
companion: [2026-07-02-roundtrip-grpo-training.md, 2026-07-02-codesign-next-stage.md, 2026-07-02-d1-prototype-implementation.md, ../research/adverserial-RL.md, ../specs/benchmarks.md]
---

# Surrogate-reward GRPO training of the substitutor (D2 · Way 1 · v0)

## STATUS 2026-07-03 — ground truth fixed (fact recall on `out_final`), gate re-passed; spec update to benchmarks.md pending

The gate is implemented (`src/cloak/train/{reward,probes}.py`, `scripts/surrogate_validation.py`;
results in `results/surrogate_validation.json`) and ran twice:

1. **τ-axis run:** realized ROUGE-L is flat within ~0.003 across τ ∈ [.005, .5] on **all three
   corpora** — the clinical flatness finding extends to email. τ produces no utility spread to
   rank, so the τ-based gate design was vacuous; redesigned to constructed arms.
2. **Constructed-arms run** (no_privacy / τ-walk / all_floor / suppression; 16 docs × 3 corpora;
   ROUGE-L **and** BERTScore ground truths): per-doc arm-Spearman **0.02–0.20** (no-go as
   scored) — but the failure is *upstream of the surrogate*: the ground truth itself fails the
   sanity ordering. Clinical ranks **no_privacy worst** (0.2182; suppression 0.2348); enron ranks
   suppression 2nd of 4 with total spread 0.03 ≈ noise; BERTScore no better (per-doc ρ ≈ 0).
   Only **aeslc** orders the arms sanely (0.469 > 0.387 ≈ 0.391 > 0.353) — and there the
   surrogate's arm-mean ordering **matches exactly** (0.744 > 0.625 = 0.625 > 0.607).

**Blocking finding:** reference-overlap metrics on note generation (clinical) and free-form
replies (enron) are **content-blind** — they cannot distinguish the working pipeline from total
content destruction (`[REDACTED]`-everything). This blocks training *and* any utility Pareto on
these corpora, independent of the surrogate.

**Root cause (defined 2026-07-03, measured):** the headline utility scores whole-output sequence
similarity, but the mechanism can only touch the detected spans, and their words cover just
**6.4% (clinical) / 8.0% (enron) / 15% (aeslc)** of the gold's tokens — so even total destruction
of every substitutable span moves ROUGE-L by less than the per-doc generation noise (mean per-doc
arm spread 0.06–0.07). The instrument's sensitive fraction is an order of magnitude below its
noise floor; everything else observed is a symptom of that one fact:

- **Shuffled-gold noise-floor test** (control generations scored against a random other doc's
  gold): enron matched 0.1245 vs shuffled 0.0893 (**1.39×** floor — the metric barely knows which
  document it is scoring); clinical 0.2182 vs 0.0742 (2.94×); aeslc 0.4693 vs 0.0581 (**8.07×**).
  Exactly the corpus ordering of the gate's sanity outcome.
- **Per-doc arm ordering is a coin flip where the fraction is small:** suppression ≥ no_privacy in
  10/16 clinical and 7/16 enron docs; on aeslc no_privacy wins 15/16.
- **The BERTScore/ROUGE-L "gap" is two broken instruments, not a contradiction:** `score.py` calls
  `bert_score` without `rescale_with_baseline=True`, so scores compress into the well-known
  ~[0.83, 0.90] fluent-English band — observed range 0.833–0.849 across everything from
  no_privacy to all-`[REDACTED]`. An uncalibrated scale next to a noise-dominated one.
- **τ-flatness is the same cause:** τ re-levels spans inside that ~6–8% token share; the metric
  cannot resolve it.
- **Secondary aggravators (real, not the cause):** enron golds are the historically-sent replies,
  not derivable from the email (reply depends on context outside the input — irreducible reference
  entropy, cf. control ROUGE-L 0.1245); clinical generations are truncated by `max_tokens=512`
  (~285 words vs ~570-word golds), a constant recall haircut on all arms.

The surrogate gate's ρ ≈ 0.02–0.20 is therefore **uninformative about the surrogate** (correlation
against noise is ≈ 0 by construction); on the one corpus whose ground truth has signal (aeslc) the
surrogate's arm ordering matches exactly. The proposed fix below is the root-cause fix precisely
because gold-fact recall on `out_final` concentrates all metric mass on the perturbable facts —
sensitive fraction ≈ 1 by construction.

Secondary findings absorbed: `u_qa` is the only content-sensitive surrogate component; `u_nli` is
~flat on register-shifted real text and has a systematic coarsening bias (R-generalized
hypotheses are logically weaker → easier to entail) — demote or redesign; probe coverage even
with fuzzy matching is 0.2–0.9/doc vs the 5–10 assumed (restatement-based probe supply is thin).

**Fix implemented + gate re-run (2026-07-03):** realized ground truth = **gold-fact recall on
`out_final`** (`reward.fact_recall` — the same probes, reader pointed at the final output; no
generalization/inversion, `out_final` is original-space). Probe supply was the second blocker and
had its own root cause: `restated_probes` misapplied `_is_role_phrase` (a lowercase-PERSON
heuristic whose `[a-z]+` tokenization turns "November 7" into the WordNet noun "november") to
every R entry, eating ~60% of unique surfaces; plus the sentence splitter broke on "Dr." and
spoken-vs-written variants ("40 milligrams"/"40 mg"). Fixed → probes/doc 0.88→**3.19** clinical,
0.69→1.0 enron; aeslc stays 0.38 (subject-line golds genuinely restate little — corpus reality).

Re-run outcome (16 docs × 4 arms × 3 corpora, round trips cached, `results/surrogate_validation.json`):

- **Fact recall orders the arms sanely on all three corpora** (no_privacy clearly first
  everywhere; suppression last/tied-low) — the same run where ROUGE-L ranked no_privacy *worst*
  on clinical. The upstream ground-truth blocker is resolved.
- **Per-doc arm-Spearman vs the surrogate:** `factrecall~u_qa` = **0.367** clinical (n=12),
  **0.44** aeslc (n=5), **0.775** enron (n=8) — vs 0.02–0.24 against ROUGE-L. `u_surr` (with
  `u_nli` mixed in) is worse than `u_qa` alone on clinical (0.183 vs 0.367) — confirms the
  demote-`u_nli` finding; the surrogate's utility term should be `u_qa`.
- **New first-class finding (report, don't engineer around):** realized fact recall of the
  working pipeline (tau_walk 0.012 clinical / 0.038 enron) ≈ all_floor ≈ suppression, ≪
  no_privacy (0.129 / 0.183) — the current round trip destroys the probed facts nearly as
  thoroughly as `[REDACTED]`-everything. Channel: `gen_absent` dominates inversion totals — the
  remote model doesn't repeat generalized phrases verbatim, so rule inversion can't restore the
  original surfaces into `out_final`. This is the real utility cost the old metrics were blind
  to, and it is what training (and/or the D-extractor upgrade) must attack.

Residual gate caveats: aeslc has probe-bearing docs on only 5/16 (thin gold restatement); the
spec change (fact recall as headline utility) still needs to land in
[`benchmarks.md`](../specs/benchmarks.md).

**Scope (decision 2026-07-02).** This plan is the active v0 of Way 1: the substitutor cascade is
trained against a **model-free local surrogate reward** — a "mini round trip" through a local QA
reader, the D1 extractor, and an NLI encoder — instead of the remote-LLM round trip. Motivation:
the round-trip reward is expensive (a remote generation per rollout candidate) and binds training
to one frozen remote model (`Qwen3.6-35B-A3B`) whose quirks the policy would learn
(reward overoptimization — background: [`adverserial-RL.md`](../research/adverserial-RL.md)).
The **true round trip remains the evaluation**, and the full round-trip-reward design — mechanism,
forks, wall-time — stays intact in
[`2026-07-02-roundtrip-grpo-training.md`](2026-07-02-roundtrip-grpo-training.md) for future
revisit; the upgrade criteria are at the end of this doc.

## Definitions

- **Surrogate (proxy) reward:** a cheap local stand-in for the true reward oracle; here
  `r = α·(1−A(doc_p)) + (1−α)·U_surr(doc_p, R)` with no generation model anywhere in the loop.
- **Gold output:** the task reference the corpus supplies (clinical SOAP note / email reply or
  subject) — by the benchmark selection rule it *restates the substituted spans*, which is what
  makes both probes below constructible without a teacher.
- **Restated-span probes:** the R spans whose original surface appears in the gold output — the
  facts the task provably must carry; probe = type-templated question (or cloze), gold answer =
  the original surface. Rule-built by intersecting R with gold (word-boundary → rapidfuzz).
- **Proposition set `P_d`:** the gold output's sentences, R-generalized — "does the information
  this note/reply sentence needs survive in `doc_p`?".
- **`invert_R(·)`:** rung-A rule alignment mapping a span of `doc_p` back to its original surface
  via the substitution record R.
- **`U_QA` (answerability):** mean token-F1 of the local extractive reader's answers from
  `doc_p`, *after* R-inversion, against the original golds.
- **`U_NLI` (premise retention):** fraction of R-generalized propositions that `doc_p` entails.
- **Mini round trip:** reader-answers-from-`doc_p` → extractor-inverts — the surrogate's local
  analog of `RemoteLLM → extractor`.
- GRPO, policy, α, `A`, round trip, R, τ-walk, MTI: see the
  [round-trip plan's definitions](2026-07-02-roundtrip-grpo-training.md#definitions).

## Decisions imported from the round-trip plan (2026-07-02 grilling; unchanged here)

1. **Policy scope — staged:** stage 1 trains the lattice-level **ranker** alone as a contextual
   bandit against the frozen greedy infiller; stage 2 unfreezes the **infiller** and trains both
   with the shared group-relative advantage (MAGRPO/MMOA-RAG pattern).
2. **RL tooling — hand-rolled:** ~150–250-line GRPO loop on `AutoModelForSeq2SeqLM` (no
   maintained framework supports enc-dec; vLLM enc-dec broken on ROCm). Stage 1 needs even less:
   REINFORCE over a per-span level softmax, plain torch.
3. **Attack head `A` = MTI probe** with the correlation-gated upgrade rule (2026-07-02 survey: no
   off-the-shelf document-level head exists; the SynthPAI encoder head is the specified escalation).
4. **Reward form = scalar α mix**, swept across runs for the Pareto curve; lexicographic kept as
   the documented fallback. **Why a sweep:** α is this method's privacy knob (analog of τ / ε),
   but it acts at *training time* — the trade-off is baked into the weights, so one trained policy
   = one point on the privacy–utility plane, and each further point costs a retrain (τ and ε are
   inference-time knobs; α is not). Matched-realized-privacy comparison needs a curve, hence ~3 α
   values → 3 runs. If the runs land clustered in realized privacy, α is a bad placement knob —
   that's the lexicographic fallback's trigger.

## Component initialization and the policy boundary (decided 2026-07-02)

### Ranker init — decided: frozen MTI roberta + MLP head (option 1)

Never RL from random (SFT-first is the imported decision; a random policy makes all G rollouts
noise). No published model performs lattice-level selection (head-survey pattern), so the fork was
the feature backbone:

- **(1) Decided — frozen roberta-base (the MTI probe's model) as feature extractor + trainable
  MLP head** over (span-in-context embedding, candidate-level embedding, MTI risk of that level,
  span type, level depth, task family). Frozen backbone → features precompute and cache once per
  corpus: the bandit trains on cached vectors in seconds, the α sweep re-runs nearly free, no
  second backbone in memory. Init: behavior-clone the τ-walk (labels auto-generated by running it).
  Known ceiling: frozen generic features may not encode task-relevance — accepted, and covered by
  the ladder below.
- *(3, ablation floor — run alongside, ~50 lines on the same features)*: feature-only logistic
  head (no text embedding). The (1)−(3) gap measures the value of contextual features; (3) alone
  learning per-(type, task) thresholds already exceeds a single global τ.
- **(2) Pre-registered follow-up — dedicated finetuned encoder** (DeBERTa-v3-xsmall/small,
  22–140M; document with span marked + candidate level as input, all weights train in SFT and RL).
  *Strengths:* the highest ceiling — representations adapt to the decision itself (e.g. "in
  dialogue→note, dates/meds are load-bearing; in email replies, org names are"), which is exactly
  the task-aware selection H2 bets on; a null result is informative, not ambiguous. *Costs:* a
  second backbone in GPU memory next to the reward scorers; every epoch re-encodes every
  (span, level) pair with current weights, so nothing caches — ~10× wall-clock (≈5–7 h stage 1,
  ≈8–12 h stage 2 per α sweep, vs <1 h / 3–5 h for option 1), and the gap widens linearly with
  corpus size and every reward-shaping iteration.

**Why option 1 was decided (2026-07-02):** v0's question is *whether* round-trip-aware selection
learning shifts the frontier at all — not its ceiling. Option 1 answers that in ~10–15 min per α
run, which makes the genuinely untested parts (surrogate reward shaping, probe quality, α
placement) cheap to iterate on; under option 2 every reward bug or mis-placed α costs an
overnight-scale rerun before it is even noticed. Option 1's known ceiling is accepted because the
ladder disambiguates every outcome: if it shifts the frontier, the mechanism is proven at 1/10th
the cost and (2) becomes a targeted ceiling measurement; if it nulls, the (1)−(3) gap plus error
analysis says whether context features carried signal frozen weights couldn't exploit — the
trigger for paying (2)'s price exactly once, on a question the cheap runs have already sharpened.

**Infiller init (unchanged, decided in D1):** flan-t5-base (span infilling is its pretraining
objective) + LoRA; SFT-distilled from the teacher lattice cache (`data/lattice_cache.json`),
AbsPyramid/AbsInstruct augmentation if the cache is thin. Generative role → pretrained generator
is the only sane start; enters at stage 2 already SFT'd.

### Policy boundary — the detector stays frozen (rationale, previously undocumented)

The detector (GLiNER small-v2.1 zero-shot ∪ Presidio rules ∪ noise filters + alias coref) contains
a trainable model, but it must **not** join the RL policy under the current reward: the utility
term rises when spans stay specific and the privacy term (MTI) only scores spans that *were
substituted* — a missed span costs the reward nothing, so gradients would teach the detector to
**under-detect**. A reward-hacking channel, not a variance problem. Additionally, detection has
gold supervision (TAB) — supervised GLiNER finetuning is the cheaper, safer fix for the measured
QUASI gap (0.857; DEM 0.56 / MISC 0.21 / QUANTITY 0.25), and detection recall remains the
*measured privacy ceiling* reported per project philosophy (RD2). **Revisit trigger:** if the
attack head escalates to the document-level encoder (fork 1 escalation), unmasked PII becomes
priced by the reward and detector-in-the-loop becomes a legitimate future fork.

## Training corpora — the task-oriented benchmarks (2026-07-02 task-eval session)

Training and utility ground use the task corpora
([`benchmarks.md`](../specs/benchmarks.md); loaders `cloak/corpora.py`, scoring `cloak/score.py`),
selected so the **gold output restates the substituted spans** — inversion fires and coarsening
carries a reference-scored cost (the old prefix/summarization smoke left inversion unexercised,
gen_absent 128/128):

- **Clinical dialogue→note (primary):** ACI-Bench (67, full SOAP) + MTS-Dialog (200, section
  notes) — notes restate age/sex/meds/dates, exactly the quasi spans lattices coarsen. Wiki:
  [yim2023_acibench](../../research-wiki/papers/yim2023_acibench_visit_note_generation.md)
  ([arXiv 2306.02022](https://arxiv.org/abs/2306.02022)),
  [benabacha2023_mtsdialog](../../research-wiki/papers/benabacha2023_mtsdialog_clinical_note.md)
  ([ACL 2023.eacl-main.168](https://aclanthology.org/2023.eacl-main.168/)).
- **Email, real PII:** AESLC subject-line (200; light restatement) + Enron reply (200; names
  people/orgs — the direct-placeholder stress test de-identified clinical notes can't give). Wiki:
  [zhang2019_aeslc](../../research-wiki/papers/zhang2019_aeslc_subject_line_generation.md)
  ([arXiv 1906.03497](https://arxiv.org/abs/1906.03497)).
- **SynthPAI stays on the attacker axis only** (8-attribute inference at evaluation), not the
  utility/training ground.
- **Doc split per corpus:** gold outputs feed the training reward (probe construction) on the
  train split only; held-out docs are evaluation.

**Regime caveat (measured, clinical τ-sweep 2026-07-02):** on clinical docs ~all generalized spans
sit at most-specific even at τ=.005 — MTI guess-back barely bites, so the reward's privacy term
gives the ranker little gradient there; the email corpora (and the eval attacker) supply the
privacy pressure. Train on the mix, report per-corpus, never merged.

## Module layout (decided 2026-07-02)

Training code lives in **`src/cloak/train/`** — a subpackage of the method package, so the
deployed inference path (`cloak/*.py`) stays visibly free of training-only code (gold references
never touch runtime modules): `train/reward.py` (surrogate `U_QA`+`U_NLI`), `train/features.py`
(frozen-roberta feature cache), `train/ranker.py` (MLP policy + bandit), `train/grpo.py` (stage-2
seq2seq loop). Runner scripts stay in `scripts/` (`surrogate_validation.py`, …); artifact names
use method names, never plan ids.

## The surrogate reward — a local mini round trip

### Per document, once (Phase 0)

1. **Restated-span probes** — spans rule-selected from R ∩ gold; the probe *question* is
   teacher-written (one short natural question per span, local llama-swap, cached to
   `data/surrogate_probes.json` — `cloak/train/probes.py`). *Measured design change
   (2026-07-02):* the original no-teacher cloze phrasing is out-of-distribution for SQuAD2
   readers — the reader abstains on every cloze probe (null beats best span by ~6 logits) while
   natural questions extract cleanly (+10-logit margins); the one-time cached teacher call keeps
   the per-candidate reward loop model-free.
2. **Propositions** `P_d` — sentence-split the gold output, R-generalize each sentence (teacher
   fallback only if sentence-level NLI proves too coarse).
3. **Scorers loaded once:** extractive QA reader (SQuAD-tuned DeBERTa-class, ~180M) + the NLI
   encoder already used as the lattice truthfulness gate (~180M). Both batch on the iGPU.
4. **Probe split:** hold out a random subset of probes/propositions per document — training reward
   uses only the train split; the held-out split is scored at evaluation (anti-Goodhart, see Risks).

### Per rollout candidate `doc_p⁽ⁱ⁾` (batched)

1. Read: `â_j = QA(q_j, doc_p⁽ⁱ⁾)` (span or "no answer").
2. Invert: `â_j' = invert_R(â_j)` via `R⁽ⁱ⁾` (rules; the real extractor participates in every
   reward evaluation).
3. `U_QA = mean_j F1(â_j', a_j)`; "no answer"/unalignable → 0.
4. Generalize propositions through `R⁽ⁱ⁾` → `p̃_l`; `U_NLI = (1/m)·Σ_l 1[doc_p⁽ⁱ⁾ ⊨ p̃_l]`.
5. `U_surr = ½(U_QA + U_NLI)`; `r⁽ⁱ⁾ = α·(1−A(doc_p⁽ⁱ⁾)) + (1−α)·U_surr⁽ⁱ⁾`.

### Why the shape is right (the R-inversion invariance)

Because answers are inverted through R before scoring, **legitimate invertible coarsening costs
nothing** ("Oslo"→"a Norwegian city" still scores F1 = 1 after inversion), so utility does not
reward under-anonymization — keeping "Oslo" earns no extra utility; only the privacy term
differentiates (the AgentStealth surface-similarity pathology is structurally excluded). Utility
falls exactly for **destructive** substitutions: answers the reader can't locate, spans R can't
align back, propositions the document no longer entails, deletions that leave no anchor
(NaPaRe's anti-extractor lesson, priced into the reward). The surrogate measures
**extraction-compatible information preservation**.

## Training loops (identical structure to the round-trip plan, reward swapped)

- **Stage 1 — ranker bandit (per doc, per epoch):** sample G level-assignments from
  `π_θ(ℓ|s,d,task)` (infiller frozen greedy) → assemble `doc_p⁽ⁱ⁾`+`R⁽ⁱ⁾` → surrogate reward →
  group advantage `adv⁽ⁱ⁾ = (r⁽ⁱ⁾−mean)/std` → REINFORCE update of θ only (+KL to SFT init).
  No remote calls, no caching machinery; G and epochs are no longer metered by reward cost.
- **Stage 2 — joint GRPO:** additionally sample infills `y⁽ⁱ⁾ ~ p_φ` (log-probs stored at sampling
  time), same shared scalar advantage on both loss terms (ranker action log-probs + infiller token
  log-probs, PPO-clipped, `KL(p_φ‖p_SFT)` leash), LoRA update of φ.
- Full step-by-step of both loops: [round-trip plan → Mechanism](2026-07-02-roundtrip-grpo-training.md#mechanism--step-by-step)
  (only its Phase-1 steps 2–3, the remote round trip, are replaced by the surrogate above).

## Surrogate validation — before any training run

Cheap sanity gate (subsumes the round-trip plan's fork 2). *Design revised 2026-07-03 after the
first run:* the original τ-axis correlation is vacuous — realized utility is flat in τ on all
corpora — so the gate tests **constructed arms** with guaranteed quality spread per doc
(no_privacy / τ-walk / all_floor / suppression) and scores the mean per-doc Spearman between the
realized and surrogate orderings of the arms. **Go:** clearly positive rank agreement where the
ground truth itself orders the arms sanely. **No-go:** disagreement → fix probes/scorers or fall
back to the round-trip plan. **Current outcome (2026-07-03 re-run, fact-recall ground truth):
positive on all three corpora — see STATUS at top** (`factrecall~u_qa` 0.37/0.44/0.775 on
clinical/aeslc/enron; arm ordering sane everywhere; use `u_qa` as the surrogate utility term,
`u_nli` demoted).

## Evaluation protocol (unchanged — the true round trip)

- Trained policy's operating points (α sweep) are evaluated with the **real round trip**
  (`Qwen3.6-35B-A3B`) on **held-out docs of the task corpora** — utility = ROUGE-L/BERTScore of
  `out_final` vs gold (`cloak/score.py`) — and the **held-out frontier-LLM attacker** on `doc_p`
  and `out_final` (leak-through; SynthPAI attribute inference + email/TAB-style entity recovery),
  at matched realized privacy vs the τ-walk Pareto — per the parent plan's shared protocol, plus
  the held-out probe split from Phase 0.
- **Second-remote-model arm** (D1 plan): re-evaluate the chosen points under a different task
  model — the measured answer to "does the policy generalize beyond one frozen model", which the
  surrogate makes more plausible (nothing in training saw Qwen) but does not guarantee.
- **The surrogate-vs-round-trip gap is a first-class result:** if surrogate-trained ≈
  round-trip-trained (when/if the latter is ever run) or simply shifts the frontier on real-round-trip
  eval, the expensive oracle was unnecessary — reportable either way.

## Wall-time estimate (v0.1 scale: ~150 train docs — ≈50 clinical + ≈100 email — G=8, ~5 epochs, 3 α)

Training is now **GPU-bound, not proxy-bound**; the perf gate's job shifts to batch saturation.

- **One-off:** probe/proposition construction is rule-based (R ∩ gold, sentence split) — seconds;
  no teacher calls. Infiller SFT (stage 2 only) = rung-B LoRA, 2–4 h if not already trained.
- **Stage 1:** 150×8×5 = 6,000 candidates per α × (k≈5–10 reader + m≈5 NLI + MTI forwards) ≈ 10⁵
  batched encoder forwards + trivial bandit updates → **well under 1 h**; a single α run in
  ~10–15 min.
- **Stage 2:** 6,000 flan-t5 sample+backward units per α (short spans, batched) + same reward cost
  → **~1–2 h per α, ~3–5 h for the sweep**.
- **Evaluation (remote, once per sweep):** 3 α points × ~100 held-out docs × (round trip +
  attacker calls) ≈ low thousands of proxy calls ≈ **~1–2 h**, cached.
- Contrast with the round-trip plan's estimate (half a day + 1–2 days, proxy-metered): roughly an
  **order of magnitude cheaper**, and iteration on reward shaping no longer costs remote calls.

## Risks

- **Probe-coverage Goodharting** — utility exists only where probes look; the policy can protect
  probe-relevant spans and degrade unprobed content. Mitigations: probe diversity across task
  families, the held-out probe split scored at eval, and the frontier attacker on `out_final`.
- **Scorer weakness = reward noise** — a small reader failing on generalized phrasing penalizes
  legitimate coarsening. Watch the stage-1 reward variance; the validation gate catches gross
  mis-shaping.
- **Blind to phrasing-induced remote failures** — placeholder-confused or specificity-starved
  remote generations are invisible to the surrogate; this is the residual the evaluation gap
  measures, and the trigger for the round-trip revisit (below).
- **Still a frozen proxy** — the Gao overoptimization playbook applies (KL leash, held-out eval
  models, low optimization pressure); see [`adverserial-RL.md`](../research/adverserial-RL.md).
- **Reward hacking through R** — unchanged from the round-trip plan: audit R's information
  content; leak-through check on `out_final` stays mandatory.
- Naming rule: code/artifacts use method names (`surrogate_reward_*`, `ranker_bandit_*`), never
  plan ids.

## Kill / upgrade criteria

- **Validation no-go:** `U_surr` fails to rank the D1 τ points like realized utility → fix or
  abandon the surrogate before training (report the disagreement).
- **Kill:** surrogate-trained policy shows no frontier shift over the τ-walk at matched realized
  privacy on the *real-round-trip* evaluation → selection learning isn't the lever on this corpus
  (or the surrogate can't express it — disambiguated by error analysis: if failures concentrate
  in phrasing-induced remote errors the surrogate can't see, that's an upgrade signal, not a kill).
- **Upgrade to the round-trip reward** ([`2026-07-02-roundtrip-grpo-training.md`](2026-07-02-roundtrip-grpo-training.md)):
  when error analysis shows the surrogate's blind spot (remote phrasing failures) dominating the
  residual utility loss, or when multi-task/multi-model training makes the round-trip cost worth
  paying (per the 2026-07-02 session decision: round trip accepted as the future full method).

## Sources

Decisions and surveys: [`2026-07-02-roundtrip-grpo-training.md`](2026-07-02-roundtrip-grpo-training.md)
(all round-trip-reward design, forks, and the tooling/optimization surveys);
[`adverserial-RL.md`](../research/adverserial-RL.md) (frozen-model RL background, surrogate
terminology, Goodhart playbook) and the papers registered there:
[Gao et al. 2022](../../research-wiki/papers/gao2022_reward_overoptimization.md)
([arXiv 2210.10760](https://arxiv.org/abs/2210.10760)),
[s3](../../research-wiki/papers/jiang2025_s3_search_agent.md)
([arXiv 2505.14146](https://arxiv.org/abs/2505.14146)),
[MMOA-RAG](../../research-wiki/papers/chen2025_mmoa_rag.md)
([arXiv 2501.15228](https://arxiv.org/abs/2501.15228)). Method anchors:
[AgentStealth](../../research-wiki/papers/shao2025_agentstealth.md)
([arXiv 2506.22508](https://arxiv.org/abs/2506.22508)),
[NaPaRe](../../research-wiki/papers/huang2025_tree_search_rewriting.md)
([arXiv 2509.20838](https://arxiv.org/abs/2509.20838)) — deletion-is-anti-extractor, priced by
the surrogate;
[SEAL](../../research-wiki/papers/kim2025_seal_adversarial_distillation.md)
([arXiv 2506.01420](https://arxiv.org/abs/2506.01420)).

Task corpora ([`benchmarks.md`](../specs/benchmarks.md) is the living spec):
[ACI-Bench](../../research-wiki/papers/yim2023_acibench_visit_note_generation.md)
([arXiv 2306.02022](https://arxiv.org/abs/2306.02022)),
[MTS-Dialog](../../research-wiki/papers/benabacha2023_mtsdialog_clinical_note.md)
([ACL 2023.eacl-main.168](https://aclanthology.org/2023.eacl-main.168/)),
[AESLC](../../research-wiki/papers/zhang2019_aeslc_subject_line_generation.md)
([arXiv 1906.03497](https://arxiv.org/abs/1906.03497)).
