---
type: reference
status: current
created: 2026-07-05
updated: 2026-07-06
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

## TLDR

1. Behavior cloning is the base of everything — every run starts by cloning the floor-walk teacher (behavior_clone), never RL from random. That's the init, not an RL algorithm.
2. ExIt (ReST^EM-style expert iteration) is the primary optimizer per the spec's Phase 2: exit_round samples G rollouts per doc through the real round trip, keeps only rollouts that strictly beat the floor-walk baseline's realized recall, and clone_choices does SFT on the winners; repeat for --exit-rounds. It was chosen as the workhorse because it has no advantage estimation, no leash arithmetic, and is robust to the quantized reward that killed the two previous runs.
3. RLOO policy gradient is the refiner layered after ExIt: train_roundtrip does classic critic-free REINFORCE with a leave-one-out baseline (no std division), the DAPO tie-filter, an entropy bonus, KL off by default, and optionally exact per-span counterfactual credit (--cf-frac).

## Pinned components (one table = the re-gate surface)

Every row is frozen for a whole gate → train → eval cycle; changing any row invalidates the
cache, the gate verdict, and all policies trained under it (§ The one subtlety).

Concrete bindings picked 2026-07-05 for the local hardware (Strix Halo gfx1151, llama-swap
proxy at :8060; family separation across grader/teacher/environment roles):

| component                            | role                                                          | binding                                                                                                                                                                                                                                                                                                                                                    |
| ------------------------------------ | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| detector                             | spans, once, frozen into the arms artifact                    | GLiNER multidomain fine-tune (`pii_gliner_multidomain` ckpt-2479 @0.3)                                                                                                                                                                                                                                                                                     |
| lattice + NLI gate                   | generalization candidates, truthfulness-filtered              | WordNet/GeoNames sources + frozen NLI entailment gate; hygiene-audited (perplexity screen) before any run                                                                                                                                                                                                                                                  |
| `aset_count` + floors k_T            | legality mask; the only privacy knob and operating-point knob | deterministic (`cloak.anonymity`), surrogate spec §3.3-2                                                                                                                                                                                                                                                                                                   |
| π_rank                               | per-span choice from `legal[s\|k]`                            | ModernBERT-base (~150M) doc encoder + per-span action head, autoregressive over spans; trains on the iGPU in `.venv`; 19-feature vector (§ π_rank — features and how they drive the choice); feature-only MLP retained as capacity-ablation arm                                                                                                                                                                                     |
| π_fill (E1)                          | grammar-constrained rendering                                 | **TBD — decided at Stage-2 kickoff** (user decision 2026-07-05). Constraints that survive the deferral: torch-side in `.venv` (HF logits-processor grammar masks + LoRA cannot live behind llama-swap), ~1–3B trainable on the iGPU, mature HF/PEFT/TRL support. Candidate at time of writing: Qwen3-1.7B-Instruct                                         |
| BC teacher                           | π_rank init                                                   | floor-walk                                                                                                                                                                                                                                                                                                                                                 |
| remote task model                    | executes the task on doc_p; the reward's LLM                  | **gemma 4 (E4B)**, non-thinking (`enable_thinking: false`), max_tokens 512, **temp 0**, cache key `hash(model, template, doc_p)`. Go/no-go: Phase-0 ceiling-anchor pass rate per corpus — fallback Qwen3.6-35B-A3B (re-pin ⇒ re-gate)                                                                                                                      |
| task prompt                          | the SAME per-corpus template everywhere                       | `TASK_TEMPLATE[corpus]`, `src/cloak/tasks.py`                                                                                                                                                                                                                                                                                                              |
| extractor                            | `invert(out_p, R)`                                            | rule exact/fuzzy-90, deployed path                                                                                                                                                                                                                                                                                                                         |
| QA reader                            | answers probes against out_final                              | **Qwen3.5-0.8B** generative reader (re-pin 2026-07-06 from `roberta-base-squad2`: extractive abstained ~40% on relational/section-structured notes, FM1); grounded+abstaining prompt, temp0/greedy (deterministic), non-thinking, **batched per out_final** (pmap workers = `-np 6`); **served on llama-swap** (:8060, `UD-Q8_K_XL` GGUF — Qwen3.5 is hybrid-attention, its fla/causal-conv1d kernels don't build on ROCm, so llama.cpp serves it and prompt-caches the shared note prefix; must stay co-resident with the reward model to avoid swap-thrash in training). Shares no gradients. Fact scorer v2 (`fact_score`): canon + number-gate + containment + acronym. See `docs/issues/2026-07-06-placeholder-gaming-reward-qa-necessity.md` for the reader-selection sweep. |
| probe teacher + atomic decomposition | writes probe questions                                        | **Qwen3.6-35B-A3B**, non-thinking. Different family than the reward model ✓. One-time, cached; teacher-tagged cache auto-retires other-teacher entries                                                                                                                                                                                                     |
| distilled RM                         | candidate screening ONLY, never gradients                     | `qwen3-embedding-0.6b` (already served) doc_p embeddings + ridge/GBM regressor on cached (doc_p → realized recall) pairs                                                                                                                                                                                                                                   |
| eval-only                            | second remote model; frontier attacker                        | second remote = **LFM2.5-8B-A1B** (thinking, max_tokens ≥1536 — third family, uninvolved in training or probes after the teacher escalation; a reasoning remote doubles as the provider-diversity check); frontier attacker = paid frontier model, **requires explicit user approval per standing rule** (local Qwen3.6 attacker for dev-side checks only) |

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

## π_rank — features and how they drive the choice

π_rank scores every legal action of a span from a per-action feature row, then samples from a
**masked softmax over that span's legal set**; spans are decided autoregressively (left→right)
within a doc. Two interchangeable backbones share the identical feature contract and
`sample`/`log_probs` interface — a context-blind MLP and a doc-conditioned encoder (below).

### The per-action feature row (`action_features`, `cloak/train/ranker.py`)

`action_features(span, corpus, floor) → tensor[n_actions, N_FEAT]`, one row per candidate
action (lattice levels ∪ placeholder). **`N_FEAT = 7 scalars + 7 type-one-hot + 5 corpus-one-hot
= 19`** (was 17 pre-`aeslc`/`wikibio`; corpora grew 3→5).

| idx | feature | formula | encodes | context-dependent? |
|-----|---------|---------|---------|--------------------|
| 0 | is_placeholder | `1.0` iff placeholder | the drop-everything action | no |
| 1 | walk_risk | `a["walk_risk"]` | contrastive re-id risk of the fill *in its sentence* | **yes** (slated for removal — see note) |
| 2 | p6 | `fill_proximity(fill,orig)` = cos_MiniLM | fill↔original semantic closeness | no (context-blind by construction) |
| 3 | level_index | `min(i,4)/4` | the action's depth in the lattice (0 = most specific) | no |
| 4 | n_levels | `min(n_lvl,4)/4` | how many level actions this span offers | no |
| 5 | log10_aset | `log10(max(aset,1))/9` | anonymity-set size of the fill | no |
| 6 | log10_active_floor | `log10(max(floor,1))/9` | the per-type operating floor (privacy knob; fed so the policy can condition on it under floor-randomization) | no |
| 7–13 | type one-hot | DEM·DATETIME·LOC·QUANTITY·ORG·MISC·OTHER | the span's PII type | no |
| 14–18 | corpus one-hot | clinical·enron·aeslc·lexsum·wikibio | the source corpus | no (doc-level, not span) |

Scalars 0–5 and the one-hots are precomputed offline into the arms artifact at env-build
(`build_arms_artifact.action_table`); the trainer reads cached values and adds only
`log10_active_floor` at load (it depends on the run's floor).

### How a choice is made (inference, per span)

```python
# spans decided left→right; each span's distribution is over its legal set only
feats = action_features(span, corpus, floor)      # [n_actions, N_FEAT]
legal = [i for i,a in enumerate(span.actions)      # legality mask = the ONLY privacy gate
         if a.mode=="placeholder" or a.aset >= floor[type]]
scores = MLP(feats)[legal]                         # feature-only backbone
#        head(cat[ctx_emb(span), feats])[legal]    # encoder backbone (below)
logp   = log_softmax(scores)                       # normalized over legal actions only
action = argmax(logp) if greedy else sample(logp)  # illegal actions are never scored
```

The masked softmax **is** the inference step: **legality (the floor) decides *what may be
chosen*; the learned scores decide *which of the safe options*.** Privacy is enforced before the
policy sees the menu — π_rank is a utility maximizer over an already-safe set, never the privacy
mechanism.

Trace (LOC span "Oslo", clinical, floor k=100): candidate levels {"a Norwegian city" aset 60,
"a city in Europe" aset 4000} ∪ {placeholder}. Mask: aset 60 < 100 → illegal; menu =
[`a city in Europe`, `placeholder`]. Head scores those two → softmax → e.g. `[0.82, 0.18]` →
pick "a city in Europe" (keeps utility; the floor already guaranteed privacy via the mask).

### Two backbones — and how the encoder interacts with the ranker

Both implement the same `sample`/`log_probs` contract; `--policy {mlp|encoder}` selects one.

| | feature-only MLP (`RankerPolicy`) | doc-encoder (`EncoderPolicy`, default) |
|-|-----------------------------------|----------------------------------------|
| head input | `feats` | `cat[ctx_emb ; feats]` |
| context channel | **none** | frozen ModernBERT-base CLS of the ±256-char span window |
| trained params | the MLP | the head **only** (encoder frozen) |
| role | capacity-ablation floor | the doc-conditioned policy |

Encoder wiring (`EncoderPolicy`) — it does **not** replace the ranker, it feeds it:
- `embed_contexts` runs the **frozen** encoder ONCE per span at load (`span_context` = ±256 chars
  around the span; `ctx_emb = last_hidden_state[:,0]`, the CLS token) and caches the vector. The
  encoder never runs inside the RL loop and takes no gradient.
- `log_probs` concatenates that one per-span vector onto **every** action row:
  `x = cat[ctx_emb ; feats[legal]] → head(x) → masked softmax`. So the context vector is
  **per-span, action-agnostic** — it shifts the span's whole distribution and interacts with the
  per-action features only through the head's learned cross-terms.
- The KL reference (`clone_for_ref`) **shares the same frozen encoder object** and deep-copies
  only the head; policy and ref differ solely in head weights.

Because the encoder is frozen + precomputed, both backbones cost the same to optimize; the
encoder's only price is load-time embedding + VRAM residency.

### How `ctx_emb` changes the choice (why a per-span-constant vector is not inert)

`ctx_emb` is identical across all of a span's actions, so it cannot re-rank them in a *linear*
scorer — it would add the same term to every action's score and cancel in the softmax. It works
only through the head's **ReLU nonlinearity**. Head = `Linear(768+19→128) → ReLU → Linear(128→128)
→ ReLU → Linear(128→1)`. For one span (`ctx` fixed, `feats_a` per action):

```
W1·[ctx ; feats_a] + b1  =  (W1_ctx·ctx)  +  (W1_feat·feats_a + b1)      # first layer splits
                             └─── g ───┘      └──── per-action ────┘
h_a = relu( g + W1_feat·feats_a + b1 )                                   # g = per-span bias
```

`g = W1_ctx·ctx` is the same 128-vector for every action. The ReLU kink is where it bites: a
hidden unit driven negative by `g` stays **off (0)** regardless of `feats_a`; one driven positive
**passes `feats_a` through**. So `g` **selects which hidden units are live → which linear
combination of the action features scores this span.** The map `feats_a → score_a` is therefore a
*different function per context*: identical action features (same aset, same p6) can score high
under one context and low under another. That is how a per-span-constant vector re-ranks actions —
it reshapes the feature→score function, it does not add to scores. (Same route for type/corpus/
walk_risk; all are span-constant and all would be dead weight in a linear ranker.)

Flip example — a span with legal levels `L_specific` / `L_coarse` (+ placeholder), same aset
profile: when `ctx` encodes "this fact is task-relevant", `g` activates units that up-weight
specificity → `score(L_specific) > score(L_coarse)` → keep the specific fill (utility); when `ctx`
encodes "not task-relevant", different units fire where specificity carries no score →
`score(L_coarse) ≥ score(L_specific)` → generalize harder (free privacy headroom). The
**utility reward** teaches `g`'s unit-selection: rollouts where specificity raised fact-recall
reinforce the `ctx → specificity-boost` path.

This is a *capacity*, not a guarantee — it requires the frozen CLS to actually encode
span-task-relevance AND the head to learn the mapping; neither is measured (no ablation; v1/v2
never left BC init). A raw MLM CLS may not expose task-relevance linearly, which is why a
contrastively-trained embedder (+ mean-pool) is the upgrade path if the encoder underperforms.

### The context-awareness invariant (critical design surface)

Exactly **two** feature channels depend on *where the span sits in the document*: **walk_risk**
(sentence-level) and the **encoder CLS** (window-level). Every other feature — `aset`, `p6`,
`level_index`, `n_levels`, type, corpus — is a function of the fill and type alone, context-free.
Consequences:

- π_rank *needs* context to choose well: which floor-legal generalization best preserves
  **utility** depends on the surrounding text (does the downstream task still recover the fact?).
- Under the **utility-only reward**, the useful context is *utility*-context, learned by the head
  through the reward gradient — exactly what the encoder supplies (general, trainable-through-the-
  head), not what walk_risk supplies (a fixed *privacy*-flavored scalar the utility reward cannot
  gradient on).
- Therefore `EncoderPolicy` is the context mechanism; `RankerPolicy` is context-blind by
  construction (identical choice for any span sharing (type, aset-profile, corpus)). **Dropping
  walk_risk makes the encoder the *sole* context channel** — which raises, not lowers, the bar on
  validating it (no ablation has run; the frozen raw-CLS representation is unproven, and a
  contrastively-trained embedder + mean-pool is the principled upgrade if it underperforms).

### Note — walk_risk as a policy feature (status: live in code, slated for removal before the pilot)

walk_risk was a deliberate feature under the *old* reward
([structural-lattice-risk plan](../../plans/2026-07-04-structural-lattice-risk.md), Task 4 kept
it; the migration's cleanup grep whitelisted "a feature/diagnostic — none on a deployment
decision path"). It carries within-span level-ordering signal and the one thing anonymity counts
are blind to: context-dependent re-identifiability (the famous-context gap of `aset`). The
third-revision **utility-only** reward orphaned it: privacy is floors-only (the gate, not the
policy, enforces it) so its privacy content is ungradable, and its utility-relevant content is
near-collinear with `log10_aset` / `level_index` / `p6` and subsumed by the encoder's context.
It also drags the pythia-410m contrastive probe into **per-request arms-build at deployment**
(arms-building is per-request for a novel doc), against the layer's "efficient, local"
positioning — an unpriced cost, since "offline-only" held for the *legality path* but not the
*feature path*. Decision (2026-07-05): **cut it as a policy feature before the pilot**
(`action_features` index 1; N_FEAT 19→18; the arms artifact keeps its stored `walk_risk` as the
offline diagnostic already designated). Optional confirmation: train the pilot both ways and
report the (expected-null) delta. *Until that edit lands, index 1 above is live.*

### Note — corpus one-hot (slated for removal, deployment generality)

The corpus one-hot (idx 14–18) is a **train/deploy skew** feature: at deployment the layer
receives an arbitrary user document with **no corpus label**, so the one-hot cannot be filled
without asking the user, auto-classifying, or feeding zeros (off-distribution from training). It
exists only to let one shared policy specialize across the five *training* corpora; domain flavor
is already carried implicitly by the encoder, which generalizes to unseen domains — the project's
open-label-generality requirement. It is legitimate for the in-corpus pilot (eval is per-corpus on
these five) but must go before any open-domain deployment claim. Decision: **remove** (idx 14–18;
N_FEAT 19 → 14). If walk_risk (idx 1) is also cut, N_FEAT → 13 (6 scalars + 7 type-one-hot); the
shared policy is then domain-agnostic, relying entirely on the encoder for context — which raises
the bar on validating the encoder (see the context-awareness invariant above).

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
