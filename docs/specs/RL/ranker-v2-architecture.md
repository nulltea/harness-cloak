---
type: reference
status: partial
created: 2026-07-22
updated: 2026-07-22
tags: [rl, ranker, model-architecture, policy, context, privacy-utility,
       bioclinical-modernbert, candidate-pair-encoding]
companion: [docs/specs/RL/ranker-v2-architecture-decision-log.md,
            docs/specs/RL/interactive-ranker-v2.md]
---

# Ranker v2 architecture — semantic privacy-head prototype

**Status: selected prototype design; not yet a production architecture.** This specification
defines the first architecture to prototype for the metadata-shortcut and policy-state-aliasing
problems. It fixes the semantic privacy head, additive lambda controller, and frozen bidirectional
candidate-conditioned context readout. Candidate features come from an ordered joint encoding of
runtime type, source, and candidate through the same frozen clinical encoder. The direct-count
factorized policy remains the stricter, more auditable privacy-formal alternative in the companion
decision log.

## Executive decision

Ranker v2 learns two different semantic judgments:

1. a utility tower judges whether a candidate preserves task-relevant document meaning;
2. a semantic privacy head predicts a log-count distribution from the source-to-candidate
   abstraction relation, then deterministic profile-menu normalization produces the policy's
   privacy score.

The actor does not receive true count, authored level position, number of levels, lambda, or
predicted privacy as generic features. Lambda combines the separately produced utility and privacy
scores through a fixed explicit additive controller:

```text
z(s, a, lambda) = u_theta(s, a) + alpha * g(lambda) * p_hat_profile(a)
```

This design intentionally asks the model to generalize semantic abstraction to unseen candidate
wording. Each matched profile's own frozen `level_counts` supervise log-count magnitude,
within-profile ordering, and profile-relative privacy progress, but counts are not required as
deployed action inputs.

Utility context is candidate-specific. A frozen bidirectional encoder produces a cached token bank
for the complete document; a trainable query derived from the source-to-candidate relation attends
over target, local, repeated-occurrence, and long-range evidence before the utility tower scores the
action.

For the selected semantic prototype, this specification overrides the type-normalized count score
in `interactive-ranker-v2.md`: its exact local privacy objective uses the frozen profile-relative
target defined below. The retained direct-count fallback continues to use the earlier strict
type-normalized score.

## Goals

- Force utility decisions to depend on document context and candidate meaning rather than count or
  lattice position.
- Learn an inspectable semantic privacy prediction that can transfer to unseen profiles and level
  wording.
- Make every source region observable to the utility policy and let each candidate retrieve the
  evidence relevant to its own semantic preservation.
- Preserve one checkpoint with a finite ordered lambda menu.
- Keep lambda-zero behavior exactly equal to the semantic utility policy.
- Keep the architecture simple enough to falsify before adding deeper context encoders or sequence
  models.

## Non-goals

- The semantic privacy prediction is not realized privacy, a formal anonymity guarantee, or a
  replacement for held-out re-identification evaluation.
- The first prototype does not re-encode a candidate-rendered or partially rewritten document at
  every decision. It uses candidate-query attention over frozen `doc_orig` token states plus
  candidate-conditioned cross-attention over explicit selected-action utility memory.
- The prototype does not infer missing training targets. Every supervised lattice level requires
  an explicit admitted count target and provenance; model-proposed targets remain experimental and
  are reported separately.
- The first prototype does not fine-tune the frozen text encoder.

## Architecture

### Data flow

```text
doc_orig --> overlapping chunks --> frozen bidirectional encoder --> cached token bank H_doc
occurrence offsets -------------------------------------------------> target/position features M_j

runtime type -------------------+
source surface -----------------+--> frozen shared pair encoder --> fixed r_pair(j, a)
candidate wording --------------+                                 |                |
                                                                    |                |
                                                                    v                v
                                                          utility projection   privacy projection
                                                                    |                |
H_doc + M_j -----------------------------+                          |                |
r_U(j, a) --> candidate query -----------+--> trained attention --> c(j, a)          |
selected-action utility memory ----------+             |                            |
                                                       v                            v
                                                utility tower                 mu_logK, sigma_logK
                                                       |                            |
                                                       v                  profile-menu normalization
                                                    u(j, a)                         |
                                                       |                     p_hat_profile(j, a)
                                                       +-------------+--------------+
                                                                     |
supported lambda --> fixed g(lambda) --> global nonnegative alpha ---+
                                                                     |
                                                                     v
                                                           combined action logit
                                                                     |
                                                            dynamic legal mask
                                                                     |
                                                                   softmax
```

The frozen relation encoder is shared computation only. Separate trainable utility and privacy
projections prevent branch-specific supervision from modifying the other branch. If a later design
unfreezes any encoder parameters, it must use separate adapters or explicit gradient isolation.

The first prototype uses one immutable bidirectional encoder checkpoint and tokenizer for both
relation strings and document chunks. The two paths have separate cached inputs and separate
trainable projections; only the frozen base parameters are shared. Using different relation and
document encoders is a later optimization fork, not part of the prototype.

### Pinned encoder checkpoint

The first prototype fixes the shared encoder and tokenizer to:

```text
model: thomas-sounack/BioClinical-ModernBERT-base
revision: c3648aa87af95837c809e6f0c5f85d08160db437
architecture: ModernBERT base, 150M parameters, hidden size 768
remote model code: disabled
```

Both relation strings and document chunks must load this exact revision. Substituting
`BioClinical-ModernBERT-large`, another clinical encoder, or a newer `main` revision creates a
different experiment and invalidates frozen-state caches.

The encoder's pretraining corpus includes all 207 ACI-BENCH documents. The current 67-document ACI
artifact may therefore support prototype development, RL training, and pipeline diagnostics, but
it must not be used to select this encoder, compare it with another encoder, or claim
out-of-corpus generalization. Encoder selection and any generalization claim require a non-ACI
clinical validation corpus whose documents were not included in encoder pretraining.

### Frozen document token bank

The prototype tokenizes the complete `doc_orig` into overlapping inputs for a pinned bidirectional
encoder:

```text
encoder input length: 512 tokens, including model special tokens
source-token overlap: 64 tokens
document truncation: none for the frozen prototype corpus
```

Every source token must appear in at least one chunk. Offset mappings preserve the relation between
source character spans, tokenizer pieces, chunks, and frozen occurrence IDs. Chunk boundary
duplication is bookkeeping, not repeated semantic evidence: aggregation tracks source-token
identity so duplicated overlap tokens cannot receive double weight merely because they occur in two
chunks.

Frozen token states are persisted separately from trainable attention outputs. Cache identity
includes:

```text
environment hash
document/source hash
encoder model and immutable revision
tokenizer and immutable revision
chunk length and overlap
offset-mapping format version
```

The first prototype encodes the complete current corpus rather than retrieving a subset of chunks.
Longer-than-supported deployment documents require a separately approved budget/retrieval policy;
the benchmark adapter must not silently truncate them.

### Target and position features

For decision `j`, augment cached token states with trainable embeddings derived from frozen,
deterministic token features:

```text
token belongs to a current-decision occurrence
token belongs to another controlled occurrence
ordinary token
relative-position bucket to the nearest current-decision occurrence
source section/chunk position
```

Every occurrence mapped to the current decision is marked. Relative features are based on original
source positions, not duplicated chunk-local positions. QA assertion IDs and dependency IDs are
never policy inputs; they remain training and evaluation supervision only.

### Candidate-conditioned context readout

The utility projection of the source-to-candidate relation produces an action query:

```text
q(j, a) = W_query r_U(j, a)
```

The query attends over the complete augmented document token bank:

```text
c_global(j, a) = Attention(q(j, a), H_doc + M_j)
```

The readout also retains target-anchored evidence:

```text
e_occ(j, i) = pooled frozen states for occurrence i of decision j
c_target(j, a) = Attention(q(j, a), {e_occ(j, i) + position(j, i)})
c_local(j, a) = candidate-conditioned attention over target-containing chunks
c_context(j, a) = Project([c_target(j, a); c_local(j, a); c_global(j, a)])
```

Repeated target occurrences are aggregated by candidate-conditioned attention with their position
features; they are not unconditionally averaged. The target summary ensures that global attention
cannot ignore the controlled surface, while local and global summaries expose nearby and distant
relations.

The frozen base encoder is not trained. Candidate query, target/position embeddings, attention,
occurrence aggregation, and context projection are trainable. Candidate-rendered re-encoding,
LoRA, partial unfreezing, and full encoder fine-tuning are excluded from the first prototype and
require new logged decisions after the frozen readout is evaluated.

### Semantic relation input

Every action is represented relative to its source:

```text
[TYPE] health-condition
[SOURCE] kidney transplant
[CANDIDATE] solid organ transplant
```

The shared relation encoder uses runtime type, canonical source surface, and rendered candidate
wording in one ordered encoder input. It does not encode the candidate alone and does not encode
source and candidate independently in the selected architecture. The serialization uses ordinary
text field labels and line separators; it does not add randomly initialized special tokens to the
frozen tokenizer.

The tokenizer records masks for the type, source, and candidate fields. One joint frozen-encoder
forward pass produces contextualized token states, then deterministic masked means produce
field-aware summaries:

```text
e_type(j, a) = Mean(H_pair[type tokens])
e_source(j, a) = Mean(H_pair[source tokens])
e_candidate(j, a) = Mean(H_pair[candidate tokens])

r_pair(j, a) = Concat([
    e_type,
    e_source,
    e_candidate,
    e_candidate - e_source,
    e_source * e_candidate
])
```

Although the summaries are pooled by field, every token state is conditioned on all three fields
through the joint bidirectional encoder. Ordered concatenation and the signed difference make
`source -> candidate` distinguishable from `candidate -> source`; the elementwise product retains
shared semantic content. `r_pair` itself has no trainable parameters; the pinned BioClinical
ModernBERT encoder remains frozen.

Separate trainable projections derive `r_U(j, a)` and `r_P(j, a)` directly from `r_pair(j, a)`.
The utility projection supplies the candidate query for document attention and the utility tower.
The privacy projection supplies only the semantic privacy head. Utility losses update only the
utility projection; privacy/count losses update only the privacy projection. Neither branch can
modify the other branch's features or the frozen encoder.

The relation cache key includes the encoder revision, tokenizer revision, serialization version,
runtime type, canonical source, rendered candidate, and action mode. KEEP, level, and placeholder
actions all traverse the same relation path; learned free-standing KEEP or placeholder embeddings
are not permitted.

The privacy branch may additionally receive a frozen categorical count-basis/source-family token
after the pair encoder, with an explicit unknown value when absent. This token never enters the
utility projection and never contains a numeric count or universe size. Optional ontology
definitions are excluded initially because availability and quality differ across profiles. They
may be added only as a separately evaluated artifact-backed input.

Action rendering is symmetric:

| Action mode | Relation candidate text | Privacy target |
|---|---|---:|
| KEEP | canonical source surface | exact `0` |
| lattice level | rendered level wording | log-count distribution learned from own-profile `level_counts` |
| placeholder | type-specific placeholder description | exact `1` |

KEEP and placeholder still receive semantic relation embeddings for utility scoring. Their privacy
scores are fixed endpoints rather than learned predictions.

### Shared-pair encoder fitness spike

Run this bounded spike before full hybrid RL. Its purpose is to test whether a frozen MLM encoder
plus separate trainable branch projections exposes enough ordered abstraction information for
utility and privacy learning. It does not select the base encoder; that checkpoint is already
pinned. It selects whether the shared-pair representation is fit for the ranker or must be
augmented.

#### Comparison arms

Use the same frozen BioClinical ModernBERT revision, downstream head parameter budget, data split,
optimizer budget, and seeds for all primary arms:

1. **Selected joint pair:** one ordered type/source/candidate input with field-aware pooling and
   `r_pair` as specified above.
2. **Candidate-only baseline:** the legacy candidate-text pooled vector without source interaction.
3. **Independent bi-encoder baseline:** separately encode source and candidate, then apply the same
   ordered algebraic pair features used by the selected arm.

The comparison isolates the value of joint cross-field self-attention from domain pretraining,
candidate semantics, and ordered pair algebra. BioLORD is not part of the primary spike; it is the
first conditional escalation if the selected representation fails in the specific ways below.

#### Diagnostic data

Construct a cached, auditable relation set from grounded profiles with real `level_counts` only.
Split by complete profile so no source surface, candidate menu, or count trajectory from a held-out
profile appears in training. Preserve runtime-type coverage in every split and report results by
type as well as in aggregate.

Each held-out profile contributes, where available:

```text
KEEP / exact relation
legal source-to-generalization relations at every grounded level
reversed candidate-to-source relations
type-matched related but non-substitutable hard negatives
type-matched unrelated negatives
placeholder endpoint
```

Hard negatives require manual or artifact-backed validation; lexical proximity alone cannot label
substitutability. Contextual utility cases must include local, long-range, and multi-decision QA-v2
assertions. ACI cases may be reported as development diagnostics, but encoder-representation
fitness and generalization conclusions require a non-ACI clinical validation slice because the
pinned encoder saw ACI-BENCH during pretraining.

#### Measurements

Train only the separate utility/privacy projections and small diagnostic heads. Measure:

1. **Direction and relation:** held-out-profile discrimination of legal generalization, reversal,
   related-but-non-substitutable, and unrelated pairs; report reversal and hard-negative confusion
   separately.
2. **Privacy structure:** held-out-profile log-count distribution loss, within-menu privacy-order
   accuracy, rank correlation with grounded profile-relative log counts, and calibration by
   runtime type.
3. **Utility relevance:** held-out prediction of QA-v2 candidate outcomes and pairwise ranking of a
   utility-preserving candidate over a context-breaking candidate.
4. **Action sensitivity:** candidate swaps must change the learned utility query and downstream
   outcome when the candidates differ semantically; source/candidate reversal must not be treated
   as an equivalent input.
5. **Operational cost:** relation-cache build time, bytes per unique relation, peak memory, and
   incremental latency relative to the candidate-only baseline.

Geometry such as cosine distance may diagnose collapse but cannot pass the spike by itself. The
load-bearing evidence is held-out outcome prediction and ordering.

#### Promotion rule

Pre-register numeric thresholds after the diagnostic set's class balance and label reliability are
measured. At minimum, the selected joint-pair arm must consistently outperform both baselines
across seeds on direction/hard-negative discrimination and contextual utility ranking, while not
regressing held-out privacy ordering or violating the one-encoder operational budget. A gain on
seen profiles without a gain on profile-held-out data is a failure.

Failure blocks promotion to full hybrid RL and triggers error attribution before adding capacity.
Reader failures, noisy counts, invalid hard negatives, or context-attention failures must be fixed
at their source rather than attributed to candidate representation.

#### BioLORD escalation trigger

[`FremyCompany/BioLORD-2023`](https://huggingface.co/FremyCompany/BioLORD-2023) is the first
candidate-feature escalation only when failures concentrate in concept semantics or hierarchy:

- legal parent/generalization relations are confused with reversals or siblings;
- related but non-substitutable concepts repeatedly receive the same relation judgment;
- within-profile privacy ordering fails on held-out medical profiles despite grounded consistent
  counts;
- unseen-profile errors are materially larger for ontology-rich clinical concepts than for simple
  lexical abstractions;
- utility errors persist for category-preserving candidates even when the required context is
  local and correctly retrieved.

Do not trigger BioLORD for long-range context misses, QA-reader instability, count-grounding
problems, lambda-controller errors, or user-defined concepts outside its ontology coverage.

If triggered, add frozen BioLORD ordered pair features to the failing branch only. Use a
privacy-only auxiliary arm for direction, hierarchy, or privacy-order failures; use a utility-only
auxiliary arm for local category-preservation failures after context retrieval has been verified.
Do not add BioLORD to both branches in the initial ablation and do not replace the document encoder.
Compare against the selected pair encoder under identical profile-held-out splits and head budgets,
and complete a license/ontology-coverage review before running the arm. Promoting BioLORD beyond a
branch-isolated auxiliary feature requires a new decision-log entry.

### Selected-action utility memory

The prototype represents earlier decisions as an explicit memory of the utility semantics that
were selected. It does not compress the prefix into a recurrent hidden state. For each prior
decision `k < j` with selected action `a_k`, construct one memory row:

```text
m_k = Project_hist([
    r_U(k, a_k),
    action_mode(k, a_k),
    runtime_type(k),
    source_position_pool(k)
])

M_<j = {m_k : k < j}
```

`source_position_pool(k)` is a deterministic decision-level pooling of the original coordinates of
all occurrences mapped to decision `k`. It carries no selection-step index and does not duplicate a
decision merely because its surface repeats.

For each current candidate `a`, use its utility relation and retrieved document context to query
the selected-action memory:

```text
q_hist(j, a) = W_hist_q [r_U(j, a); c_context(j, a)]
h_hist(j, a) = CrossAttention(q_hist(j, a), M_<j, M_<j>)
```

`h_hist(j, a)` is candidate-specific: different candidates may retrieve different prior choices.
The first prototype uses one trainable cross-attention block, not a recurrent network, memory
self-attention stack, or full Transformer. An empty memory returns the exact zero vector.

The memory contains no count, predicted privacy, lambda, authored level index, menu size, profile
identity, QA assertion identifier, or QA dependency identifier. Count and privacy losses cannot
update the memory projection, query projection, or cross-attention parameters. The legal-mask
state, including fill-collision constraints, remains external to semantic memory.

Memory rows form a set of selected semantic actions, not a record of arbitrary sampling order.
Therefore the architecture adds no selection-step positional embedding. It retains the original
source/occurrence position and may derive query-relative source distance from it. Permuting memory
rows while preserving their contents must leave `h_hist` and the action logits unchanged, within
numeric tolerance.

This representation is intentionally limited to prior selected actions. It does not expose future
actions to early decisions and does not reconstruct the partially rewritten document. If validated
failures require bidirectional coupling across the complete action vector, reopen joint or two-pass
decision inference. If they require literal effects of prior rendered substitutions rather than
their selected semantic relation, reopen partially rendered document encoding. Neither failure is
a reason to restore count-bearing recurrent state.

Decisions are processed in deterministic first-occurrence walk order, matching
`interactive-ranker-v2.md`. The prototype does not randomize this order during training.
Development diagnostics replay deterministic reverse and seeded alternative orders while
preserving dynamic legality; material order sensitivity reopens a two-pass draft-and-refine design.

#### History-module fitness spike

Before promotion, compare three matched-budget arms on the same data splits and seeds:

1. no action history;
2. a corrected utility-only GRU receiving the same count-blind selected-action records;
3. the selected candidate-conditioned cross-attention memory.

Report complete utility on validated multi-decision and counterfactual cases, utility on local and
single-decision cases, runtime, and memory. Add targeted interventions that change one relevant or
irrelevant prior action. The selected module must react to the relevant change, remain stable under
the irrelevant change, preserve memory-row permutation invariance, and retain useful early actions
on long documents. Current ACI artifacts contain only 4--38 decisions per document (median 11), so
quadratic attention over selected rows is not a performance justification for recurrence.

Promote selected-action cross-attention only if it improves validated multi-decision utility over
no history without regressing local cases or violating the invariance and selectivity tests. If it
does not, remove action history. Do not silently fall back to the GRU.

### Utility tower

The utility tower receives:

```text
context representation C(document, decision, action)
utility projection of relation embedding r_U(source, candidate, type)
explicit context-candidate interaction
action mode and runtime type
candidate-specific selected-action context h_hist(j, a)
```

It does not receive:

```text
raw or normalized count
predicted privacy score
authored level index
number of levels
lambda magnitude or profile identity
count provenance
```

`C(document, decision, action)` is the candidate-specific `c_context(j, a)` defined above, not a
shared decision vector. Its output `u_theta(s, a)` is an action utility logit. It need not be
calibrated as an absolute expected utility value for this prototype. Utility-only ExIt, document
utility advantages, and counterfactual utility comparisons update this tower.

The selected-action cross-attention block is the normative history interface. The context token
bank represents `doc_orig`; selected-action memory represents earlier choices without exposing
privacy shortcuts. The legacy GRU and its count-bearing `decision_action_inputs` are not retained.

### Semantic privacy head

The privacy branch receives the privacy projection of the source-to-candidate relation and, when
available, a frozen categorical count-basis/source-family token. It does not receive document
context, task assertions, selected-action memory, lambda, authored position, menu size, raw count,
numeric universe size, or stable action identity as an embedding.

For every lattice level, the head predicts a distribution over log count:

```text
mu_raw(a), sigma_raw(a) = privacy_head(r_P(source, candidate, type, count_basis))
mu_logK(a) = softplus(mu_raw(a))
sigma_logK(a) = sigma_min + softplus(sigma_raw(a))

log(K(a)) ~ Normal(mu_logK(a), sigma_logK(a)^2)
```

The target is the matched profile's own raw count:

```text
ell_j(a) = log(max(level_counts_j[a], 1))
```

`level_counts` is a profile-specific count source, not a normalization rule. The prototype never
replaces it with a global fill-string lookup that aggregates equal wording across profiles.

The controller score is derived jointly across the current profile's lattice-level menu:

```text
denom_hat_j = max_b_in_profile_levels mu_logK(b)
p_hat_profile,j(a) = clip(mu_logK(a) / denom_hat_j, 0, 1)

p_hat_profile,j(KEEP) = 0
p_hat_profile,j(placeholder) = 1
```

The corresponding frozen training target uses `ell_j` in the same formula. Profiles with two or
more levels whose true denominator is zero are flat and fail the privacy-head training gate. A
one-level profile assigns its sole level score one, including when its count is one, and is tagged
`singleton_profile_normalization`; singleton results are reported separately because they contain
no within-profile ranking signal.

The privacy loss combines distributional count calibration, within-profile ordering, and direct
calibration of the score consumed by the controller:

```text
L_privacy = L_logcount_nll(ell, mu_logK, sigma_logK)
            + rho * L_pairwise_rank(mu_logK, ell)
            + gamma * L_huber(p_hat_profile, p_profile)
```

Pairwise terms exclude tied targets. The likelihood preserves approximate magnitude and supplies
an uncertainty estimate; ranking prevents good average count fit from hiding incorrect menu
ordering; profile-relative calibration verifies the actual policy input.

Training and reporting stratify targets by grounding status and source family. Experimental
model-proposed counts may supervise the prototype only when explicit and admitted by the count
artifact; they never become formal privacy labels. The head is judged separately on grounded and
model-proposed subsets.

For audit, report the geometric median estimate and log-space interval:

```text
K_hat(a) = round(exp(mu_logK(a)))
interval_95_lower(a) = max(1, exp(mu_logK(a) - 1.96 * sigma_logK(a)))
interval_95_upper(a) = exp(mu_logK(a) + 1.96 * sigma_logK(a))
```

`K_hat` is always labelled model-predicted. Integer rendering does not turn it into a sourced,
grounded, or certifying count.

The controller uses only `mu_logK` through `p_hat_profile`. Predicted uncertainty is an audit,
calibration, and abstention diagnostic; the first prototype does not reward uncertainty or subtract
it from privacy pressure.

### Explicit additive lambda controller

Lambda does not condition either semantic branch. It enters only after the utility tower and
semantic privacy head have produced lambda-invariant scores:

```text
b(a, lambda) = alpha * g(lambda) * p_hat_profile(a)

alpha = softplus(alpha_raw)
g(lambda) = log1p(lambda) / log1p(max_lambda)
```

The finite supported lambda menu and its positive maximum are frozen run state. `g` is a fixed
deterministic transform, not a learned network. Lambda profile identity is not embedded or passed
to the model; the supported numeric value completely determines controller strength.

`alpha` is the controller's only trainable parameter. It is one global scalar shared across every
document, action, runtime type, corpus, and supported lambda value. Initialize `alpha` to one when
hybrid training begins, parameterize it through `softplus`, and freeze it in the checkpoint before
held-out evaluation. Do not calibrate it separately by profile, type, corpus, method, or evaluation
set.

```text
alpha_raw_initial = log(expm1(1))
softplus(alpha_raw_initial) = 1
```

Therefore:

```text
b(a, 0) = 0
z(s, a, 0) = u_theta(s, a)
```

The prototype does not use full-representation FiLM, privacy-only FiLM, learned profile embeddings,
per-profile slopes, or separate profile heads. If one global scale cannot realize useful distinct
operating points, that finding reopens the architecture fork and requires a new logged decision;
implementation must not silently add controller capacity.

For a fixed state and legal menu, increasing lambda must not decrease expected predicted privacy.
This local monotonicity follows from the scalar additive form and is verified numerically. Whole-
document monotonicity remains an empirical diagnostic because earlier actions can change later
legal menus.

## Training protocol

### Privacy-head pretraining

Train the privacy projection and head before policy optimization. Split by complete profile, not
random action, so validation measures transfer to unseen source/candidate lattices. KEEP and
placeholder endpoints are excluded from learned-head metrics because their scores are fixed.

The head must beat all of:

- authored-position-only prediction;
- action-mode and runtime-type prediction;
- stable profile/action memorization;
- candidate-only encoding without the source surface.

### Utility-only initialization

Freeze the validated privacy head and set the controller contribution to zero. Train the utility
tower with the approved utility-only ExIt initialization, document utility credit, and available
one-decision counterfactual utility comparisons. The frozen document encoder remains in evaluation
mode; candidate query, target/position embeddings, attention pooling, context projection, utility
memory projection and attention, and utility head are trainable.

### Hybrid RL

Sample one supported numeric lambda value per document episode. Forward policy logits use the
frozen privacy prediction:

```text
z = u_theta + alpha * g(lambda) * stop_gradient(p_hat_profile)
```

Gradient ownership is explicit:

```text
utility and counterfactual losses --> utility tower, selected-action memory, controller scale
exact profile-relative objective --> controller scale only
privacy calibration/ranking       --> privacy projection and head only
```

The semantic privacy head remains frozen during the first hybrid prototype. This prevents the
exact profile-relative objective from turning `p_hat_profile` into an opaque policy preference.
The globally shared controller scale receives both utility and exact-count gradients: their
opposition determines a finite privacy--utility tilt. Sending only privacy gradient to the scale
would make saturated privacy pressure its unconstrained optimum.
Fine-tuning it during RL is a later ablation and must retain distribution, calibration, and ranking
supervision plus held-out regression gates.

Lambda-zero episodes remain in every hybrid training schedule. Their combined logits must remain
bitwise equal to utility logits before masking.

## Required diagnostics and gates

### Semantic privacy transfer

On profile-held-out levels, report:

- log-count negative log likelihood, log error, multiplicative error, and interval coverage;
- within-menu pairwise ordering accuracy and rank correlation;
- profile-relative calibration error and selected-action profile-relative regret;
- results by runtime type, count provenance, and source family;
- performance on candidate paraphrases and lexical counterexamples where apparent wording
  broadness disagrees with numeric count.

No pooled metric may hide a failing runtime type.

### Context observability and use

Compare under the same frozen encoder, document split, utility/counterfactual artifacts, trainable
parameter budget, and rollout budget:

```text
current local CLS plus occurrence mean
bidirectional local target-span pooling
full-document candidate-conditioned attention
```

Report utility ordering/regret separately for local, long-range, repeated-occurrence, and
multi-decision assertions. The selected architecture must pass:

- context-swap cases where candidate metadata is fixed but required evidence changes;
- candidate-swap cases where document state is fixed but candidate meaning changes;
- target-ablation tests showing that removing target markers degrades relevant decisions;
- full-document ablation showing value beyond target-containing local chunks;
- no material regression on local-only assertions;
- frozen-cache size, build time, attention memory, and per-document inference latency.

The previous context-injection ablation based on marginal surface recall at floor/ceiling extremes
does not resolve this fork and is not reused as selection evidence.

### Shortcut controls

The complete policy must pass:

1. **Count isolation:** changing true count metadata after privacy-head inference cannot change
   logits.
2. **Position isolation:** changing authored indices or menu size cannot change logits.
3. **Lambda-zero identity:** changing lambda/count artifacts cannot change lambda-zero utility
   logits.
4. **Context sensitivity:** with source, candidate, and privacy prediction fixed, changing relevant
   document context changes utility ordering on validated counterfactual cases.
5. **Candidate sensitivity:** with metadata and context fixed, changing candidate meaning changes
   utility and semantic privacy predictions in the expected directions.

### Gradient isolation

Tests must establish:

```text
grad(L_utility, privacy_head_parameters) = 0
grad(L_count, utility_parameters) = 0
grad(L_privacy, utility_parameters) = 0
grad(L_count, history_parameters) = 0
grad(L_privacy, history_parameters) = 0
grad(L_utility, alpha_raw) may be nonzero
grad(L_count, alpha_raw) may be nonzero
```

### Deployment contract

The deployed policy requires the pinned
`thomas-sounack/BioClinical-ModernBERT-base@c3648aa87af95837c809e6f0c5f85d08160db437`
checkpoint for both the semantic relation encoder and document encoder, plus its tokenizer,
chunking/offset configuration, relation-serialization and field-pooling versions, target-feature
schema, runtime type, canonical source surface, candidate wording, numeric lambda menu, fixed `g`
transform, selected-action-memory schema and source-position encoding, and the global `alpha` used
for evaluation. It also requires the complete candidate menu because profile-relative
normalization is set-valued. It does not require level counts as actor inputs. When counts are
available, they remain diagnostics and may be used to audit the predicted log-count distribution;
they do not silently replace the semantic prediction.

## Alternative retained for escalation

The direct-count factorized policy replaces `p_hat_profile(a)` with a strict type-normalized count
and keeps the same explicit controller:

```text
K_j(a) = matched_profile.level_counts[a]
p_type,j(a) = clip(log(max(K_j(a), 1)) / log(K_ref[type(j)]), 0, 1)
z_direct(s, a, lambda) = u_theta(s, a) + alpha * g(lambda) * p_type,j(a)
```

It is more privacy-formal and auditable because deployed privacy pressure is determined by an
inspectable count artifact rather than model inference. That advantage requires stricter counts:
every deployed level and type reference must have complete, grounded, coherently normalized,
version-pinned count state. Model-proposed counts are not admitted to this fallback's deployed
controller. It is selected if the semantic privacy head fails profile-held-out transfer,
calibration, or lexical-counterexample gates.

## Prototype decision rule

Proceed to hybrid RL only if the semantic privacy head demonstrates profile-held-out signal beyond
position, type, candidate-only, and identity baselines. Prefer the semantic prototype when its
log-count distribution, within-profile ordering, and profile-relative scores generalize to unseen
profiles and the combined policy passes shortcut and lambda-zero gates. Fall back to strict
type-normalized direct-count factorization when exact count reliability is required or semantic
privacy prediction does not generalize.

**Crux.** Ranker v2 first predicts a log-count distribution from how the candidate abstracts the
source, converts that prediction into profile-relative privacy progress, and applies lambda through
a transparent controller; strict type-normalized grounded counts remain the auditable fallback.
