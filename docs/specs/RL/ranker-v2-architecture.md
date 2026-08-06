---
type: reference
status: current
created: 2026-07-22
updated: 2026-08-05
tags: [rl, ranker, model-architecture, policy, context, privacy-utility,
       bioclinical-modernbert, candidate-pair-encoding, lexicographic-rl,
       actor-critic, primal-dual, low-rank-adapters, freeze-policy]
companion: [docs/specs/RL/ranker-v2-architecture-decision-log.md,
            docs/specs/RL/interactive-ranker-v2.md,
            docs/research/ranker-v2-trainable-multi-objective-rl-review.md]
---

# Ranker v2 architecture — three-tier lexicographic actor-critic

**Status: selected experimental architecture; not yet production-validated.** This specification
retains the frozen clinical encoder, candidate-conditioned document readout, and selected-action
cross-attention memory from the semantic-v1 policy. It supersedes both the semantic privacy-head
plus additive controller and the later head-only residual over a completely frozen semantic stack.
The replacement is a three-tier policy-based lexicographic actor: a permanently frozen clinical
encoder substrate, a utility semantic branch trained to convergence and then frozen, and a private
trainable lexicographic semantic side path with a zero-initialized residual head. Objective critics
remain separately supervised and document utility is protected by training-only primal-dual
constraints.

The deterministic epsilon-zero selector is not a deployment component. It remains an offline
oracle/test utility only. The deployed ranker always executes one ordinary neural policy forward
per sequential decision.

## Executive decision

Ranker v2 serves one report policy: **utility-first/count-second**. Its logits are

$$
z_{\mathrm{lex}}(s_t,a;\lambda_k)=
\begin{cases}
u_\theta(s_t,a), & k=0,\\
u_\theta(s_t,a)+r_\psi(x_{\mathrm{lex}}(s_t,a),e_{\lambda_k}), & k>0.
\end{cases}
$$

The frozen utility policy is retained as an internal reference and audit path, not exposed as a
second product mode in this prototype. BC, verified ExIt, and utility-only structured RL train the
utility branch. Lexicographic RL then freezes that branch and trains only a private post-encoder
semantic side path plus the residual head to improve exact profile-relative count return while a
document-level primal-dual constraint preserves the frozen utility reference.

The lexicographic operator lives in the optimization update, not in an inference-time selector and
not in a scalar logit bonus. Count and utility advantages both update the private Tier-3 semantic
adapters and residual head. Utility targets update only the utility critic; count targets update
only the count critic.
Training-only dual variables increase the utility-gradient weight when a document violates its
utility-retention constraint. They are absent from the deployed checkpoint's forward path.

One checkpoint supports a frozen finite menu $\Lambda=\{\lambda_0,\ldots,\lambda_{K-1}\}$ with
$3\le K\le5$. The caller selects one setting for a complete task/session and each document episode.
$\lambda_0$ is the exact utility-only identity. Every $\lambda_k$ with $k>0$ activates the same
shared Tier-3 actor under a setting condition $e_{\lambda_k}$. The ordered menu maps to explicit
document-level utility slacks

$$
\tau(\lambda_0)=0,\qquad
\tau(\lambda_1)=0,\qquad
0=\tau(\lambda_1)<\tau(\lambda_2)<\cdots<\tau(\lambda_{K-1}).
$$

$\lambda_1$ therefore means strict count-second optimization at the frozen utility reference;
higher settings permit pre-registered user-facing utility loss budgets. The numeric setting never
multiplies count reward, utility reward, an advantage, or a logit. Additive `alpha`, gain heads, tie
hinges, cycle projection, gap scaling, utility-logit softcaps, and profile-sensitivity training
pressure are not part of this architecture. Historical additive-controller switch points and the
historical `0.044` reader statistic cannot define the new slack menu.

Utility context remains candidate-specific. A frozen bidirectional encoder produces a cached token bank
for the complete document; a trainable query derived from the source-to-candidate relation attends
over target, local, repeated-occurrence, and long-range evidence before the utility tower scores the
action.

The count objective consumes the frozen own-profile-relative targets defined by
`interactive-ranker-v2.md`. Counts are rewards and diagnostics, never actor input features and never
realized-privacy measurements.

## Goals

- Force utility decisions to depend on document context and candidate meaning rather than count or
  lattice position.
- Let count supply a direct behavioral gradient where utility permits it, without relabeling any
  model quantity as both utility and privacy.
- Make every source region observable to the utility policy and let each candidate retrieve the
  evidence relevant to its own semantic preservation.
- Preserve one checkpoint with an immutable utility reference branch and one served
  lambda-conditioned lexicographic branch.
- Let one checkpoint serve three to five ordered user settings without storing one model per
  setting or claiming interpolation to untrained settings.
- Keep utility-reference logits exactly equal to the selected utility checkpoint throughout
  lexicographic training while allowing the private lexicographic semantic path to adapt.
- Make utility-retention violations, count movement, dual dynamics, and every gradient edge
  auditable.
- Keep the architecture simple enough to falsify before adding deeper context encoders or sequence
  models.

## Non-goals

- Count score is not realized privacy, a formal anonymity guarantee, or a replacement for held-out
  re-identification evaluation.
- The first prototype does not re-encode a candidate-rendered or partially rewritten document at
  every decision. It uses candidate-query attention over frozen `doc_orig` token states plus
  candidate-conditioned cross-attention over explicit selected-action utility memory.
- The prototype does not infer missing count targets. Every trainable lattice level requires
  an explicit admitted count target and provenance; model-proposed targets remain experimental and
  are reported separately.
- The first prototype does not fine-tune the frozen text encoder.
- The first prototype does not claim held-out generalization, a Pareto frontier, or matched
  realized privacy from its four-document mechanism run.

## Architecture

### Data flow

```text
Tier 1 · permanently frozen clinical substrate
doc_orig + source/candidate strings + sequential action records
        -> BioClinical-ModernBERT token/relation banks, masks, raw selected-action records
                         |                                |
                         v                                v
Tier 2 · utility branch, trained first then frozen       Tier 3 · lexicographic side path
frozen post-encoder semantic maps                        frozen maps + private rank-4 deltas
        -> x_U(s,a) -> u_theta(s,a)                      -> x_lex(s,a)
lambda_k -> setting condition e_lambda -----------------> GELU-16 residual r_psi(x_lex,e_lambda)
                         |                                |
                         +-------------------------------> z_lex(s,a;lambda_k)
                                                          -> legal gather -> pi_lex(.|s,lambda_k)

stop_gradient(x_U)   -> utility critic V_U
stop_gradient(x_lex) -> count critic V_P

utility advantage -------------------------------+
count advantage ---------------------------------+--> Tier-3 adapters + residual head
document utility violation --> dual mu_{d,k} -----+
```

Tier 1 is frozen from the start. Tier 2 includes the utility relation projection,
candidate-conditioned context readout, selected-action memory, interaction features, and utility
head. BC, verified ExIt, and utility-only structured RL train Tier 2; its selected checkpoint is
then frozen bit-for-bit. Tier 3 does **not** consume only a detached final utility feature vector.
It reuses the same detached Tier-1 substrate and frozen Tier-2 maps while adding private rank-4
low-rank deltas to selected post-encoder semantic transformations. This gives lexicographic RL a
trainable semantic route without allowing count gradients to alter the utility reference.

The utility and count actor surrogates both update Tier 3 and the active positive-setting
condition. No lexicographic RL gradient reaches
Tier 1, Tier 2, either critic from the actor loss, or a critic from the other critic's loss. No
trainable parameter is shared between the utility and count critics.

The first prototype uses one immutable bidirectional encoder checkpoint and tokenizer for both
relation strings and document chunks. The two paths have separate cached inputs and separate
utility-initialization projections; only the frozen base parameters are shared. Using different relation and
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

The frozen base encoder is never trained. During utility initialization, the utility candidate
query, target/position embeddings, attention, occurrence aggregation, and context projection are
trainable; they become Tier 2 and are frozen before lexicographic RL. Tier 3 then adds private
low-rank deltas only to the selected post-encoder semantic maps. Candidate-rendered re-encoding,
encoder LoRA, partial encoder unfreezing, and full encoder fine-tuning remain excluded. The selected
post-encoder adapters are therefore not an exception to the frozen-encoder contract.

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

One utility projection derives `r_U(j, a)` from `r_pair(j, a)` and supplies the candidate query for
document attention and the utility base. It is trained only during utility initialization and then
frozen. The lexicographic side path applies its private low-rank delta to the same frozen relation
map and derives `r_lex(j, a)` without modifying `r_U`. The utility critic consumes detached
`x_U`; the count critic consumes detached `x_lex`. There is no count value or count-specific input
embedding on the deployed path.

The relation cache key includes the encoder revision, tokenizer revision, serialization version,
runtime type, canonical source, rendered candidate, and action mode. KEEP, level, and placeholder
actions all traverse the same relation path; learned free-standing KEEP or placeholder embeddings
are not permitted.

Action rendering is symmetric:

| Action mode | Relation candidate text | Count reward |
|---|---|---:|
| KEEP | canonical source surface | exact `0` |
| lattice level | rendered level wording | frozen own-profile-relative target |
| placeholder | type-specific placeholder description | exact `1` |

KEEP and placeholder still receive semantic relation embeddings for utility scoring. Count rewards
remain outside actor features.

## Historical shared-pair privacy-head fitness spike (superseded 2026-08-05)

Run this bounded spike before full hybrid RL. Its purpose is to test whether a frozen MLM encoder
plus separate trainable branch projections exposes enough ordered abstraction information for
utility and privacy learning. It does not select the base encoder; that checkpoint is already
pinned. It selects whether the shared-pair representation is fit for the ranker or must be
augmented.

### Comparison arms

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

### Diagnostic data

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

### Measurements

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

### Promotion rule

Pre-register numeric thresholds after the diagnostic set's class balance and label reliability are
measured. At minimum, the selected joint-pair arm must consistently outperform both baselines
across seeds on direction/hard-negative discrimination and contextual utility ranking, while not
regressing held-out privacy ordering or violating the one-encoder operational budget. A gain on
seen profiles without a gain on profile-held-out data is a failure.

Failure blocks promotion to full hybrid RL and triggers error attribution before adding capacity.
Reader failures, noisy counts, invalid hard negatives, or context-attention failures must be fixed
at their source rather than attributed to candidate representation.

### BioLORD escalation trigger

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
problems, preference-policy errors, or user-defined concepts outside its ontology coverage.

If triggered, add frozen BioLORD ordered pair features to the failing branch only. Use a
privacy-only auxiliary arm for direction, hierarchy, or privacy-order failures; use a utility-only
auxiliary arm for local category-preservation failures after context retrieval has been verified.
Do not add BioLORD to both branches in the initial ablation and do not replace the document encoder.
Compare against the selected pair encoder under identical profile-held-out splits and head budgets,
and complete a license/ontology-coverage review before running the arm. Promoting BioLORD beyond a
branch-isolated auxiliary feature requires a new decision-log entry.

## Selected-action utility memory

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

The memory contains no count, preference mode, dual value, authored level index, menu size, profile
identity, QA assertion identifier, or QA dependency identifier. Count losses cannot
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

### History-module fitness spike

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

## Tier 2 — utility semantic branch

The utility feature stack receives candidate-conditioned context, the source-to-candidate relation,
their explicit interaction, action mode, runtime type, and candidate-specific selected-action
memory. It excludes raw or normalized count, authored level index, number of levels, preference
mode, count provenance, and every dual variable.

The utility branch emits one feature vector and one logit per legal action:

$$
x_U(s_t,a)=F_\theta(\mathcal B(s_t,a)),
\qquad
u_\theta(s_t,a)=f_\theta(x_U(s_t,a)),
$$

where $\mathcal B$ is the detached Tier-1 substrate.

BC, utility-only ExIt, structured utility policy gradients, and counterfactual utility comparisons
train the complete utility feature stack and $f_\theta$. After utility initialization, freeze all
of those parameters and record their exact state hash. Lexicographic training cannot change
$x_U$, $u_\theta$, or the selected-action state used to replay the utility reference.

The selected-action cross-attention block remains the normative history interface. Policy state
stores canonical raw selected-action records—decision identity, chosen action identity, occurrence
positions, and frozen relation/token-bank references—from which Tier 2 and Tier 3 build their own
memory projections. This prevents the lexicographic path from depending only on the final utility
feature while preserving one dynamic legal state. The legacy GRU and count-bearing
`decision_action_inputs` remain rejected.

## Tier 3 — trainable lexicographic semantic side path

The served policy adds an action-conditioned residual to the frozen utility logits:

$$
x_{\mathrm{lex}}(s_t,a)=F_{\theta,\Delta W_\psi}(\operatorname{stopgrad}(\mathcal B(s_t,a))),
$$

$$
z_{\mathrm{lex}}(s_t,a;\lambda_k)=
\begin{cases}
u_\theta(s_t,a), & k=0,\\
u_\theta(s_t,a)+r_\psi(x_{\mathrm{lex}}(s_t,a),e_{\lambda_k}), & k>0.
\end{cases}
$$

$F_{\theta,\Delta W_\psi}$ reuses frozen Tier-2 maps and adds private rank-4 low-rank deltas only
to this allowlist:

```text
utility_projection
context_readout.token_projection
context_readout.query_projection
context_readout.target_attention.{q,k,v,out}
context_readout.local_attention.{q,k,v,out}
context_readout.global_attention.{q,k,v,out}
context_readout.context_projection
memory.record_projection
memory.query_projection
memory.cross_attention.{q,k,v,out}
```

For `nn.MultiheadAttention`, `{q,k,v,out}` names four logical maps even when PyTorch stores Q/K/V
inside one packed `in_proj_weight`. The adapter registry must expose and hash the logical maps; it
may not adapt the packed tensor as one undifferentiated block.

No delta may target the clinical encoder, frozen utility head, utility mode/type embeddings, legal
masking, or count-target code. Context/relation interaction remains deterministic. The exact module
names, dimensions, rank, and parameter count are architecture-pinned; an implementation must fail
closed if the allowlist resolves differently after a refactor.

Each delta uses zero-product initialization: one low-rank factor receives the standard small random
initialization and the output factor is exactly zero. Consequently Tier 3 initially reproduces the
frozen semantic map while retaining a reachable gradient into the zero factor. The residual head's
final map is also exactly zero, making $z_{\mathrm{lex}}=u_\theta$ at initialization.

The lambda-conditioned residual head is `lexicographic-residual-lambda-ln-gelu16-v1`:

```text
LayerNorm(feature_dim=4608, affine=False)
-> Linear(4608, 16, bias=True)
-> add learned setting embedding e_lambda[k] in R^16
-> LayerNorm(16, affine=False)
-> GELU
-> Linear(16, 1, bias=False)
```

The final linear map is initialized to exact zero. Input normalization controls feature-block
scale; pre-activation normalization prevents the saturation that collapsed the previous tanh gain
head. Width `16` is fixed for the first mechanism test; width `32` is not an automatic escalation.
Rank `4` is fixed for the semantic deltas. The architecture figure's approximately 227K trainable
parameter label is a non-binding planning estimate; the implementation must emit and pin the
derived exact count from the resolved logical target registry.

The finite lambda menu owns one trainable 16-dimensional embedding row per setting. Row zero is
initialized to exact zero, never receives a gradient, and is bypassed together with the complete
Tier-3 residual at $\lambda_0$. Positive-setting rows are ordinary Tier-3 actor parameters. The
semantic low-rank adapters are shared across settings; only the residual fusion is setting-specific.
This is a finite-menu conditioned policy, not a continuous lambda model, and it makes no claim about
interpolation to unregistered settings.

Tier 3 receives no count value, authored action position, numeric slack, or dual variable as an input.
It receives only the frozen setting identity through $e_{\lambda_k}$.
Frozen action mode and runtime-type metadata may be reused because they are semantic action
descriptors already available to the utility branch. Count affects Tier 3 only through the actor
objective computed from selected actions and frozen count rewards. It must therefore learn from
document context, candidate wording, and selection history rather than copy the reward table during
inference.

The prototype's exact profile-count artifact is a temporary reward shortcut. The final architecture
replaces it with a separately trained, frozen k-anonymity estimator behind the same reward interface.
That replacement creates a new reward pin and requires retraining, but it neither changes the lambda
conditioning contract nor exposes estimated counts to the actor forward pass.

## Objective critics

The prototype has two separately parameterized state-value baselines over permutation-invariant
pools of current legal-action features:

$$
V_U(s_t,\lambda_k;\phi), \qquad V_P(s_t,\lambda_k;\omega).
$$

The utility critic consumes `stop_gradient(x_U)` and the count critic consumes
`stop_gradient(x_lex)`. Each also receives a detached encoding of the active finite-menu setting,
applies a small action projection, mean-pools over the legal menu, and predicts one scalar. This is
required because return distributions differ by setting. They may share frozen Tier-1 substrate but
no trainable parameters. Neither critic is part of deployment.

The utility critic target is the routed utility return assigned to that rollout-decision pair:
linked plus residual utility for linked decisions, and complete-document utility for uncovered
decisions. The count critic target is exact profile-relative count return-to-go from decision $t$
through the end of the legal trajectory. Both losses use SmoothL1/Huber regression in their native
reward units. Counterfactual utility terms continue to substitute for sampled utility terms on
measured pairs; they do not train the count critic.

Critic targets and advantages are detached before actor optimization. Critic losses cannot update
Tier 1, Tier 2, Tier 3, or one another.

## Training-only dual variables

For every training document $d$ and positive setting $\lambda_k$, store one direct nonnegative
multiplier $\mu_{d,k}$, initialized to zero and projected onto $[0,\infty)$ after every dual update.
It enforces the document-level constraint

$$
J_U(\pi_{\mathrm{lex}}(\cdot\mid\lambda_k);d) \ge b_d-\tau(\lambda_k),
$$

where $b_d$ is the frozen utility-only policy's achieved expected return under a pinned rollout
manifest. $\lambda_0$ has no dual because its actor path is the frozen utility identity. The first
mechanism test uses only $\lambda_0$ and strict $\lambda_1$ with $\tau(\lambda_1)=0$. Nonzero slacks
require a separately pinned three-to-five-setting menu and may not reuse the historical `0.044`
reader statistic.

The multipliers are optimizer state, not deployed model inputs. They may differ by document during
training without creating per-document inference calibration. At held-out inference only the
learned Tier-3 actor generalizes; no dual lookup or dual network is executed.

## Served policy contract

The served/report policy has one lambda-conditioned forward contract:

$$
z_{\mathrm{lex}}(s_t,a;\lambda_k)=
\begin{cases}
u_\theta(s_t,a), & k=0,\\
u_\theta(s_t,a)+r_\psi(x_{\mathrm{lex}}(s_t,a),e_{\lambda_k}), & k>0.
\end{cases}
$$

It uses the same dynamic legal mask and canonical sequential selected-action records as the frozen
utility reference. The public ranker call requires one registered `lambda_setting_id`, held constant
for the complete task/session and document episode. The checkpoint contains Tier 1, Tier 2, Tier-3
adapters, the residual head, setting embeddings, and the exact frozen lambda-menu hash.
The frozen utility path remains callable only for reference construction, parity tests, ablations,
and audit; $\lambda_0$ reaches the same result through the public policy identity branch. Critics, duals,
utility-reference manifests, and optimizer state remain training/audit artifacts.

### User-facing lambda contract

`lambda_setting_id` is a required public policy argument. It names one entry in a checkpoint-pinned
finite menu; it is not accepted as an arbitrary float. A menu entry contains:

```yaml
lambda_setting_id: lambda-0
ordinal: 0
display_label: utility
utility_slack: 0.0
mode: utility-identity
```

The manifest uses `ranker-v2-lexicographic-lambda-menu-v1`. The historical additive
`ranker-v2-lambda-menu-v1` artifact is a different schema and must fail closed.

Positive entries use `mode: lexicographic` and carry their pre-registered
`utility_slack: tau(lambda-k)`. The menu has three to five ordered entries, exactly one identity
entry at ordinal zero, a strict positive entry with zero slack, and monotonically increasing slack
thereafter. The manifest hash is part of checkpoint identity; changing labels, order, menu size, or
slacks creates a new policy version and requires training/evaluation again.

The setting is fixed for the complete task/session by the caller and for the complete document,
rollout group, old-policy replay, counterfactual batch, and utility-reference comparison by the
trainer. It cannot change between decisions. Unsupported settings fail closed. The report must show
both the stable ID and display label; it must never present the ordinal as a calibrated privacy
guarantee.

The user chooses by tolerated task-utility loss, not by an asserted privacy percentage:

- `lambda-0`: preserve the frozen utility policy exactly; apply no learned count-second behavior;
- `lambda-1`: improve count only while satisfying the zero-slack utility constraint;
- `lambda-2` and above: permit the exact document-level utility slack printed in the menu entry so
  the actor may pursue more count return.

Product copy may describe settings as utility-preserving, strict count-second, or by their explicit
utility budget. It must not label them low/medium/high privacy until held-out attacker evaluation
establishes those realized operating points.

The first mechanism run uses a two-entry, explicitly non-deployable `scope: mechanism` manifest
containing only `lambda-0` and strict `lambda-1`. It validates the conditioned interface and strict
lexicographic update without inventing nonzero user tolerances. Promotion to a deployable checkpoint
requires a separately pre-registered three-to-five-entry `scope: deployment` menu. Selecting its
nonzero slack values is an operating-point decision based on explicit utility budgets and held-out
behavior, not on historical additive switch points.

## Lexicographic training protocol

### Utility initialization

Train BC and utility-only ExIt as before, then run utility-only structured RL until the frozen
selection rule declares the base checkpoint. Build a pinned utility-reference manifest containing
the utility-only action vectors, rollout seeds, exact utility keys, mean expected utility $b_d$, and
artifact hashes for every training document used by lexicographic optimization. Freeze the utility
feature stack and base logits before creating Tier-3 adapters, the residual head, or critics. Build
Tier 3 from the frozen allowlisted maps with zero-product adapters and verify exact initialization
parity before any lexicographic update.

### Critic warm start

Using cached utility-only and count-scored trajectories, fit $V_U$ and $V_P$ without updating either
actor branch. Split diagnostics by document. A critic is usable only if its held-out-document error
beats the corresponding train-mean baseline; failure does not authorize critic gradients into the
actor. For the four-document mechanism test, critic quality is reported but the actor's structured
LOO and counterfactual utility terms remain the authoritative utility advantages.

### Lexicographic actor update

For one positive setting $\lambda_k$ held fixed across the complete document rollout group, and for
objective $i\in\{U,P\}$, define a separately auditable PPO surrogate

$$
S_i(\psi)=\mathbb{E}_t\left[
\min\left(
\rho_t\widehat A^i_t,
\operatorname{clip}(\rho_t,1-\epsilon_{\mathrm{PPO}},1+\epsilon_{\mathrm{PPO}})
\widehat A^i_t
\right)
\right].
$$

Do not combine or standardize the two advantages before clipping. The Tier-3 actor minimizes

$$
L_{\mathrm{actor}}(d,k)
=-S_P-\operatorname{stopgrad}(\mu_{d,k})S_U
+\eta_{\mathrm{KL}}L_{\mathrm{KL}}-\beta_H H(\pi_{\mathrm{lex}}).
$$

The utility term retains the existing in-place counterfactual substitution and fixed policy-role
denominator. The count term uses selected exact return-to-go and updates the Tier-3 adapters and
residual head directly. On an exact utility tie, $\widehat A^U_t=0$, so count owns the actor update without
inventing a utility preference.

The count target and count advantage are not multiplied by $\lambda_k$ or $\tau(\lambda_k)$. The
setting changes behavior only through $e_{\lambda_k}$ and the utility constraint. $\lambda_0$
groups execute the frozen identity branch and are audit controls, not actor-training groups.

The dual violation is

$$
v_{d,k}=b_d-\tau(\lambda_k)-\widehat J_U(\pi_{\mathrm{lex}}(\cdot\mid\lambda_k);d),
$$

and its minimization loss is

$$
L_{\mathrm{dual}}=-\mu_{d,k}\operatorname{stopgrad}(v_{d,k}).
$$

Gradient descent therefore increases $\mu_d$ when utility is below its target and decreases it when
the constraint is slack. Critic updates run on the fastest timescale, Tier-3 actor updates on the
middle timescale, and dual updates on the slowest. No batch-level standard-deviation normalization
is permitted because it changes the dual's meaning.

### Gradient ownership

Tests must establish:

```text
utility-base initialization loss -> utility feature stack and utility base
utility critic loss              -> utility critic only
count critic loss                -> count critic only
utility actor surrogate          -> Tier-3 adapters + residual head + active positive setting only
count actor surrogate            -> Tier-3 adapters + residual head + active positive setting only
dual loss                        -> document-setting dual parameter only

grad(lex actor losses, Tier 1)      = 0
grad(lex actor losses, Tier 2)      = 0
grad(count losses, utility critic)  = 0
grad(utility losses, count critic)  = 0
grad(critic losses, Tier 3)         = 0
grad(dual loss, actor or critics)   = 0
grad(all losses at lambda_0, Tier 3)= 0
```

### Required mechanism gates

Before any full-corpus or held-out claim:

1. frozen utility-reference logits are bit-identical before and after lexicographic training;
2. Tier-3 initialization reproduces the frozen semantic maps and logits exactly at every setting,
   and $\lambda_0$ remains bit-identical after every update;
3. count and utility actor terms both produce finite nonzero gradients in the residual head on the
   first eligible update, in at least one adapter output factor after the residual map has moved,
   and in its paired input factor after that output factor has moved;
4. every lexicographic actor gradient is exactly zero in Tier 1 and Tier 2;
5. count gradients are exactly zero in the utility critic;
6. each final-three-snapshot lexicographic greedy document has an exact utility key no lower than
   $b_d-\tau(\lambda_k)$ for every trained positive setting;
7. at least one opportunity-bearing document has stable positive count-score separation for all
   final three snapshots;
8. no document passes only through a temporary mid-run peak followed by zero separation;
9. document-setting dual variables respond in the correct direction to synthetic and real utility
   violations without leaking across settings;
10. critics beat train-mean baselines or remain report-only and excluded from actor advantages;
11. the Tier-3 allowlist resolves to the pinned modules, rank, dimensions, and trainable-parameter
    count; no encoder or utility-head parameter is trainable;
12. no deterministic selector, evidence override, additive controller, gain head, or tie hinge is
   active in the evaluated forward path.

Passing on four cache-rich documents establishes only that the gradient architecture can express
stable utility-first/count-second behavior. It does not establish document-held-out transfer,
realized privacy, or a useful multi-setting frontier.

### Deployment and artifact contract

The architecture pin includes the frozen encoder/tokenizer revisions, Tier-1 substrate schema,
utility-base checkpoint hash, Tier-2 frozen-parameter manifest, Tier-3 adapter target allowlist,
adapter rank and initialization schema, residual-head schema, finite lambda-menu schema and hash,
setting-embedding state, canonical action-memory schema, and dynamic legal-mask version. It excludes
training-only dual values. Loading fails closed on an
additive-controller checkpoint, a head-only residual checkpoint, a changed adapter target map, or
a checkpoint whose Tier-1/Tier-2 parameters are not frozen.

Required artifacts are the frozen environment and utility assertion artifact, complete profile
count targets, frozen representation manifest, utility-base checkpoint, utility-reference
manifest, frozen lambda-menu manifest, Tier-3 architecture manifest, lexicographic checkpoint,
critic checkpoint, document-setting dual-state audit, epoch reports, and the training record written
before the run.

### Implementation boundaries

- Tier-1 substrate extraction and Tier-2 utility semantics remain owned by
  `src/cloak/ranker/semantic.py` behind typed substrate and utility-stack interfaces;
- private low-rank adapters, the Tier-3 semantic side path, residual head, served lexicographic
  distribution, and critic modules live in a focused lexicographic actor module;
- sequential state stores canonical raw selected-action records; Tier 2 and Tier 3 project those
  records independently rather than sharing mutable hidden state;
- PPO surrogates, critic losses, document constraints, and dual updates live in a focused
  lexicographic objective module;
- structured utility credit and counterfactual scheduling remain in
  `src/cloak/ranker/interactive.py` and expose typed advantages to the objective module;
- orchestration and artifact validation remain in `scripts/train_interactive_ranker.py`;
- `src/cloak/ranker/lexicographic.py` remains an offline selector/oracle utility and is prohibited
  from the deployed policy path.

## Sources

- [Trainable multi-objective RL review](../../research/ranker-v2-trainable-multi-objective-rl-review.md).
- [Lexicographic actor-critic implementation plan](../../plans/2026-08-05-ranker-v2-lexicographic-actor-critic.md).
- [Interactive ranker v2 architecture report](../../html/interactive-ranker-v2.html).
- [Lexicographic Multi-Objective Reinforcement Learning](../../../research-wiki/papers/skalse2022_lexicographic_morl.md)
  ([arXiv 2212.13769](https://arxiv.org/abs/2212.13769)).
- [Reward Constrained Policy Optimization](../../../research-wiki/papers/tessler2018_reward_constrained_policy.md)
  ([arXiv 1805.11074](https://arxiv.org/abs/1805.11074)).
- [Constrained Policy Optimization](../../../research-wiki/papers/achiam2017_constrained_policy_optimization.md)
  ([arXiv 1705.10528](https://arxiv.org/abs/1705.10528)).
- [State-Augmented Constrained Reinforcement Learning](../../../research-wiki/papers/calvofullana2021_state_augmented_constrained_rl.md)
  ([arXiv 2102.11941](https://arxiv.org/abs/2102.11941)).

## Historical additive-controller design (superseded 2026-08-05)

The remaining sections preserve the previous semantic privacy-head/additive-controller prototype
for archaeology only. They are non-normative and must not be used to implement or promote the
current three-tier lexicographic actor.

### Semantic privacy head

The privacy branch receives the privacy projection of the source-to-candidate relation. A frozen
categorical count-basis/source-family token remains a training-only ablation; any checkpoint with
that input is rejected by policy loading because deployed inference does not carry its category
indices. The policy privacy branch does not receive document
context, task assertions, selected-action memory, lambda, authored position, menu size, raw count,
numeric universe size, or stable action identity as an embedding.

For every lattice level, the fixed-sigma prototype predicts a standardized log-count mean:

```text
y(a) = (ell(a) - train_mean) / train_std
y_hat(a) = mean_head(block_normalize(r_P(source, candidate, type, count_basis)))
mu_logK(a) = train_mean + train_std * y_hat(a)
```

`y_hat` is unconstrained. Negative `mu_logK` values retain regression gradients and are clamped to
zero only when computing the controller's profile-relative normalization. The relation head
normalizes the type, source, candidate, and Hadamard-product blocks with train-only statistics and
drops the linearly redundant candidate-minus-source block. The primary head is a single linear
map. An opt-in `32`-unit GELU mean head is the only capacity escalation in the prototype.

After checkpoint selection, estimate one constant audit scale from dev residuals:

```text
sigma_fixed = clamp(RMSE_dev(ell - mu_logK), 0.3, 3.0)
```

The checkpoint and reports record `sigma_fixed`; every prediction from that checkpoint uses the
same value. Learned heteroscedastic sigma is an ablation only, not part of the prototype.

The target is the matched profile's own raw count:

```text
ell_j(a) = log(max(level_counts_j[a], 1))
```

`level_counts` is a profile-specific count source, not a normalization rule. The prototype never
replaces it with a global fill-string lookup that aggregates equal wording across profiles.

The controller score is derived jointly across the current profile's lattice-level menu:

```text
mu_controller(a) = max(mu_logK(a), 0)
denom_hat_j = max_b_in_profile_levels mu_controller(b)
p_hat_profile,j(a) = clip(mu_controller(a) / denom_hat_j, 0, 1)

p_hat_profile,j(KEEP) = 0
p_hat_profile,j(placeholder) = 1
```

The corresponding frozen training target uses `ell_j` in the same formula. Profiles with two or
more levels whose true denominator is zero are flat and fail the privacy-head training gate. A
one-level profile assigns its sole level score one, including when its count is one, and is tagged
`singleton_profile_normalization`; singleton results are reported separately because they contain
no within-profile ranking signal.

The primary privacy loss combines standardized mean regression with direct calibration of the
score consumed by the controller:

```text
L_privacy = SmoothL1(y_hat, y, beta=1.0)
            + 1.0 * L_huber(p_hat_profile, p_profile)
```

The default rank weight is `rho=0`. The registered rank ablation uses `rho=0.05` and bounded Huber
regression of predicted pair differences against true log-count differences; it never uses an
unbounded sign-only logistic objective. Tied targets are excluded. Losses are macro-averaged over
complete profiles, batches contain approximately 32 complete profiles, and losses average levels
or pairs within a decision, decisions within a profile, then profiles within a batch. Duplicate
decisions are removed only when their complete ordered menus have identical source/candidate
identities, targets, normalizations, and provenance; row-level deduplication is forbidden. All
pair, segment, singleton, profile-weight, and menu-size index structures are precomputed outside
the update loop.

Training and reporting stratify targets by grounding status and source family. Experimental
model-proposed counts may supervise the prototype only when explicit and admitted by the count
artifact; they never become formal privacy labels. The head is judged separately on grounded and
model-proposed subsets.

For audit only, report the geometric median estimate and fixed-sigma log-space interval:

```text
K_hat(a) = round(exp(mu_logK(a)))
interval_95_lower(a) = max(1, exp(mu_logK(a) - 1.96 * sigma_logK(a)))
interval_95_upper(a) = exp(mu_logK(a) + 1.96 * sigma_logK(a))
```

`K_hat` is always labelled model-predicted. Integer rendering does not turn it into a sourced,
grounded, or certifying count.

The controller uses only `mu_logK` through `p_hat_profile`. Fixed-sigma uncertainty is report-only
and the first prototype does not reward uncertainty or subtract it from privacy pressure. Claims
that `K_hat` or its interval is meaningful require a separate distributional-audit gate.

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

## Historical training protocol (superseded 2026-08-05)

### Privacy-head pretraining

Train the privacy projection and head before policy optimization. Split by complete profile, not
random action, so validation measures transfer to unseen source/candidate lattices. KEEP and
placeholder endpoints are excluded from learned-head metrics because their scores are fixed.

Train with AdamW at learning rate `3e-4`, weight decay `0.01`, gradient clipping `1.0`, and at most
500 updates. Evaluate every 10 updates with patience 10 and minimum improvement `1e-3`. Select each
seed lexicographically by profile-macro calibration error, within-menu ordering, then absolute
log-count error. Do not select on NLL.

The candidate-only arm is the competitive baseline. Every neural diagnostic arm passes through a
fixed non-trainable projection to the semantic head width and then uses the same trainable head.
Reports include both the shared-head trainable count and each arm's native linear-head count; they
never repeat features or pad trainable parameters. The semantic arm must show paired
profile-bootstrap improvement on at least one of profile-relative calibration or within-menu
ordering and be non-inferior within the preregistered margin on the other. Other references are:

- authored-position plus mode/type as an oracle ceiling with a non-inferiority margin, never an
  ordering target that must be strictly beaten;
- action-mode and runtime-type prediction as a sanity floor;
- train-profile mean prediction as a sanity floor.

The blocking policy-fitness metrics are profile-relative calibration, within-menu ordering,
selected-action regret, and lexical/semantic counterexamples. Counterexamples report `N/A` until
their set exists. NLL, interval coverage, `sigma_fixed`, and absolute log-count error are
report-only; count and interval claims require the separate distributional-audit gate.

Ordering and Spearman assign predicted log-count differences within `1e-6` to the same tie group.
Pairwise accuracy gives a predicted tie `0.5` credit and reports its rate separately. Promotion
requires at least three preregistered seeds on one frozen profile split. A one-seed artifact is an
iteration checkpoint with `NEEDS_MULTI_SEED_EVIDENCE`; it cannot produce controller `PASS`.

The `ranker-v2-semantic-privacy-v2` checkpoint binds the v2 metric and diagnostic hashes, run
protocol, seed count, counterexample-set hash, and promotion verdict. Policy and hybrid loaders
require `run_protocol=promotion`, at least three seeds, a counterexample hash, and verdict
`PROMOTE`. An explicit development-only CLI override may admit another verdict, but it never admits
a count-basis checkpoint.

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

## Historical diagnostics and gates (superseded 2026-08-05)

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

## Historical direct-count additive fallback (superseded 2026-08-05)

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

## Historical prototype decision rule (superseded 2026-08-05)

Proceed to hybrid RL only if the semantic privacy head demonstrates profile-held-out controller
signal beyond the candidate-only competitive baseline and the sanity floors. Prefer the semantic
prototype when its within-profile ordering, profile-relative scores, selected-action regret, and
lexical counterexamples generalize to unseen profiles and the combined policy passes shortcut and
lambda-zero gates. Fall back to strict
type-normalized direct-count factorization when exact count reliability is required or semantic
privacy prediction does not generalize.

**Crux.** Ranker v2 first predicts a standardized log-count mean from how the candidate abstracts
the source, converts that prediction into profile-relative privacy progress, and applies lambda
through a transparent controller; strict type-normalized grounded counts remain the auditable
fallback.
