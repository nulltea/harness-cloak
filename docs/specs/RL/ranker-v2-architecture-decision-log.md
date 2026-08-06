---
type: reference
status: current
created: 2026-07-22
updated: 2026-08-05
tags: [rl, ranker, model-architecture, decision-log, shortcut-learning,
       semantic-utility, count-shaping, lambda-conditioning, context-injection,
       attention, state-aliasing, encoder-selection, candidate-pair-encoding,
       action-history, low-rank-adapters, freeze-policy]
companion: [docs/specs/RL/ranker-v2-architecture.md,
            docs/specs/RL/interactive-ranker-v2.md,
            docs/specs/RL/interactive-ranker-v2-decision-log.md,
            docs/research/ranker-v2-trainable-multi-objective-rl-review.md]
---

# Ranker v2 architecture — design decision log

This log records consequential architecture forks for ranker v2. The companion architecture spec
defines the selected prototype while this log preserves alternatives, tradeoffs, and the evidence
required before promoting a prototype to production architecture.

## Metadata shortcut can replace semantic utility learning

**Decision.** Prototype the semantic privacy-head policy first. It withholds count, authored level
position, and menu size from the actor, then learns an explicit privacy prediction from the
source-to-candidate semantic relation. The companion architecture spec defines the prototype.

Retain the direct-count factorized policy as the more privacy-formal alternative. It uses the
known normalized count in a constrained controller rather than predicting privacy from wording.
Because count directly controls deployed action preference, that alternative requires stricter
deployment-time count completeness, grounding, normalization, and auditability than the semantic
prototype. It is the fallback if the semantic privacy head does not transfer or calibrate reliably.

### Problem

The current conditional actor sends semantic and structural information through one shared
lambda-conditioned scorer:

```text
context ------------------------------------+
candidate text -----------------------------|
mode and runtime type ----------------------|
authored level position --------------------+--> FiLM(lambda) --> MLP --> action logits
number of lattice levels -------------------|
normalized count score ---------------------|
GRU history --------------------------------+
```

The scorer receives the normalized count score that defines the exact local privacy-shaping
objective. It also receives authored level position and menu size, which are strong proxies for
that score. Nothing requires the network to use count only for privacy or context and candidate
meaning only for utility.

The common lattice shape makes a shortcut especially plausible: roughly two levels often preserve
useful meaning, followed by one or two levels that lose it. A policy can therefore map lambda to a
typical lattice position without determining which level preserves the particular document's
task-relevant context.

This induces five distinct failure modes:

1. **Middle-level collapse.** The policy selects a typical lattice quantile rather than a
   context-dependent action.
2. **Semantic feature starvation.** Count and position explain reward more cheaply, so context and
   candidate representations receive little useful gradient.
3. **False lambda adaptation.** Different profiles produce different actions, but only because the
   model learned a lambda-to-position lookup.
4. **Tail and transfer failure.** The policy fails when the utility boundary occurs unusually early
   or late, or when a new profile has a different count/level geometry.
5. **Misleading evaluation.** Aggregate reward and lambda frontiers can look healthy even when
   swapping document context barely changes the selected action.

Passing count is not intrinsically label leakage. Count is a known deployment-time property and
the ranker should not have to infer it from level wording. The failure is allowing known privacy
metadata to explain the learned semantic-utility function.

### Direction: factorized policy

Replace the monolithic lambda-conditioned scorer with a semantic utility tower and a constrained
privacy controller. They produce separate action preferences that are added before masking and
softmax:

```text
context ---------------------------+
candidate meaning -----------------|
context-candidate interaction -----+--> utility tower --> u(document, action) --+
mode and runtime type -------------|                                        |
semantic decision history ---------+                                        |
                                                                              +--> add
normalized count score ------------+                                        |    --> legal mask
lambda magnitude/profile ----------+--> privacy controller --> b(action, lambda) +    --> softmax
structural privacy state, if needed +
```

The initial utility tower does not receive normalized or raw count, authored level position,
number of levels, lambda magnitude, lambda identity, or count provenance. It receives information
needed to judge semantic utility: document context, a symmetric representation of KEEP, level and
placeholder candidates, explicit context-candidate interaction, action mode and runtime type, and
any retained semantic trajectory state.

The initial privacy controller does not receive document text, semantic context, candidate text,
or utility assertions. Its minimal form is:

```text
b(action, lambda) = alpha * g(lambda) * p_count(action)

alpha >= 0
g(0) = 0
g(lambda) is nonnegative and ordered
```

For the existing ordered magnitude, a fixed starting form is:

```text
g(lambda) = log1p(lambda) / log1p(max_lambda)
```

One global nonnegative `alpha` controls the relative logit scale. The later lambda-controller
decision in this log fixes that scalar form for the prototype: failure to express useful operating
points reopens the fork rather than automatically adding profile-specific slopes or FiLM.

The combined policy is:

```text
utility logit:  u_theta(document, action, semantic_history)
privacy tilt:   b_phi(p_count(action), lambda, privacy_state)
final logit:    z = u_theta + b_phi
policy:         softmax(mask(z))
```

At lambda zero, the architecture must satisfy the exact invariant:

```text
z(document, action, lambda=0) = u_theta(document, action)
```

#### Gradient ownership

Separate modules are insufficient if every objective updates both. Forward computation uses the
same combined logits, but gradient routing preserves branch ownership:

```text
utility loss:
    log_softmax(u_theta + stop_gradient(b_phi))
    updates utility parameters only

exact count loss:
    log_softmax(stop_gradient(u_theta) + b_phi)
    updates privacy-controller parameters only

counterfactual utility loss:
    compares u_theta(action_a) with u_theta(action_b)
    updates utility parameters only
```

The utility-only ExIt warm start trains `u_theta` before count conditioning. Hybrid training keeps
lambda-zero episodes and the gradient boundary. If a GRU remains, its initial role is semantic
utility history; it does not receive count, lambda, authored position, or menu size. The dynamic
legal mask continues to enforce hard action legality independently of recurrence.

#### Architectural contrast

```text
CURRENT
known privacy metadata + semantics + lambda
                    --> one unrestricted representation
                    --> one opaque action score

FACTORIZED
semantics           --> learned utility preference --------+
known count + lambda --> constrained privacy preference ----+--> action score
```

#### Tradeoffs

The factorized policy preserves one lambda-adaptable checkpoint and the existing stochastic RL
interface. It makes lambda-zero behavior auditable, supports new profiles with explicit counts,
and prevents count gradients from training semantic utility features.

Its main costs are branch-scale calibration and restricted interaction. A minimal additive
controller may not anticipate future privacy effects caused by dynamic action collisions. A more
expressive controller can address those effects but reopens shortcut capacity, so complexity must
be added only after a measured failure of the minimal form.

Factorization does not guarantee that the utility tower will use context. Candidate wording can
still reveal generality, and the empirical utility boundary may genuinely favor middle levels.
Context-swap, candidate-swap, metadata-only, context-free, and non-middle controls remain required.

**Crux.** Count stops being a generic predictive feature and becomes a constrained privacy
controller added to a count-blind semantic utility policy.

### Selected prototype: semantic privacy head

Remove normalized count, authored level position, and number of levels from policy inputs. Replace
implicit privacy learning inside one actor with an explicit head that predicts privacy from the
semantic relation between source and candidate:

```text
runtime type -------------------------+
source surface -----------------------+--> frozen relation encoder --> relation embedding
candidate wording --------------------+                                  |          |
                                                                          |          |
document context ---------------------+                                  |          |
semantic decision history ------------+--> utility tower --> u(s, a) ----+          |
                                                                                     |
                                      semantic privacy head --> mu_logK(a), sigma_logK(a)
                                                                          |
                                      profile-menu normalization --> p_hat_profile(a) -+
                                                                                     |
supported lambda value ------------------------> additive controller ----------------+
                                                                                     |
                                                            u + controller(p_hat_profile, lambda)
                                                                                     |
                                                                            legal mask + softmax

own-profile level_counts --> log-count supervision, profile-relative target,
                             and exact profile-relative count objective only
```

The relation encoder receives the source and candidate together, not the candidate phrase in
isolation. It can therefore learn that `solid organ transplant` is broader than `kidney
transplant`, rather than treating broadness as an absolute lexical property. Runtime type is part
of the relation because abstraction and count scales differ by domain.

The semantic privacy head predicts a distribution over `log(K)` for every lattice level. A
deterministic transform jointly normalizes the predicted means across the current profile menu to
produce a profile-relative privacy score in `[0, 1]`. KEEP and placeholder retain exact endpoint
scores zero and one. Training uses log-count likelihood, within-profile ordering, and
profile-relative calibration. Profile-held-out evaluation determines whether the head learned
transferable semantic abstraction rather than action IDs or profile-specific wording.

The utility tower receives document context and the source-to-candidate relation embedding but not
count, level position, menu size, lambda, or predicted privacy. Lambda enters only through the
explicit final controller:

```text
z(s, a, lambda) = u_theta(s, a) + alpha * g(lambda) * p_hat_profile(a)

alpha >= 0
g(0) = 0
g(lambda) is nonnegative and ordered
```

Separate utility and privacy projections follow the frozen relation encoder. Privacy losses cannot
update utility parameters, and utility losses cannot update the privacy head. The first prototype
freezes a validated privacy head during hybrid RL. Utility and exact-count objectives may both
update only the nonnegative controller scale at their shared boundary; exact-count gradients remain
barred from the utility tower. This preserves the meaning of `p_hat` without leaving the controller
scale with a privacy-only, saturation-seeking objective.

This direction has the highest learning potential because it forces the model to learn both
contextual utility and semantic abstraction, and it can score unseen candidate wording without a
deployment-time count. Its risk is prediction error: linguistic broadness, ontology membership,
and numeric anonymity count are related but not equivalent. A model-derived integer estimate is
therefore reported with uncertainty and never treated as a grounded count. The direction requires
profile-held-out log-count calibration, ordering, paraphrase, and lexical-counterexample tests
before hybrid RL.

**Crux.** Privacy stops being handed to the actor as metadata and becomes an explicit, inspectable
prediction of the source-to-candidate abstraction relation, while lambda remains a transparent
controller outside semantic utility.

#### Count sourcing and normalization clarification

`level_counts` and normalization are separate choices. Both retained architectures source a raw
level count from the matched profile's own `level_counts[level]`; neither uses a global lookup that
merges equal level strings across profiles.

The selected semantic prototype uses profile-relative normalization:

```text
ell_j(a) = log(max(K_j(a), 1))
p_profile,j(a) = ell_j(a) / max_b_in_profile_levels ell_j(b)
```

This teaches progress along the source profile's available privacy ladder and cancels inconsistent
cross-profile count scales. It does not claim that equal scores imply equal anonymity-set size
across profiles. A one-level profile assigns its sole level score one and is reported as a
singleton-normalization case.

The direct-count factorized fallback instead uses strict type normalization:

```text
p_type,j(a) = clip(log(max(K_j(a), 1)) / log(K_ref[type(j)]), 0, 1)
```

That fallback admits only complete, grounded, coherently normalized, version-pinned counts and
type references. It preserves more absolute count meaning and is therefore more privacy-formal and
auditable, but it depends on stronger count consistency than the semantic prototype.

### Required evidence before confirming the production direction

Prototype selection does not settle the production architecture. Both directions are evaluated
against the same frozen environment, utility artifact, counterfactual pool, lambda menu, rollout
budget, and held-out documents. Confirmation requires:

1. a metadata-only baseline that cannot match semantic utility performance;
2. context-swap and candidate-swap tests with action metadata held fixed;
3. a dedicated subset whose utility-optimal action is not a middle level;
4. exact lambda-zero count invariance for the factorized policy;
5. profile-held-out log-count distribution, profile-relative calibration, and count-ordering tests
   for the semantic privacy head, including paraphrases and cases where lexical broadness disagrees
   with numeric count.

## Lambda uses an explicit additive controller

**Decision.** Remove lambda conditioning from the semantic representations and fix the first
prototype to an explicit additive controller:

```text
utility tower ----------------------> u_theta(s, a) --------+
                                                            |
semantic privacy head -------------> p_hat_profile(a) ------+--> add --> legal mask --> softmax
                                                            |
supported numeric lambda --> fixed g(lambda) --> alpha -----+
```

```text
z(s, a, lambda) = u_theta(s, a) + alpha * g(lambda) * p_hat_profile(a)

alpha = softplus(alpha_raw)
g(lambda) = log1p(lambda) / log1p(max_lambda)
```

`g` is a fixed deterministic transform over the frozen finite lambda menu. The controller has one
global trainable nonnegative scale, `alpha`, shared by every document, action, runtime type, and
lambda setting. Lambda profile identity is not a model input. The supported numeric value fully
determines controller strength.

The utility tower and semantic privacy head are lambda-invariant. Lambda changes only the scalar
weight assigned to the frozen privacy prediction. At lambda zero:

```text
controller = 0
combined logits = utility logits
```

This is a permanent architectural identity, not an initialization property. The utility-only warm
start disables the controller. Hybrid RL initializes `alpha` to one. Exact privacy shaping can
update only `alpha`, while utility, entropy, and KL losses may also update `alpha` through the
ordinary policy probabilities. This opposition is required: privacy-only updates would make
unbounded privacy tilt the scale's optimum. Freeze `alpha` with the checkpoint before held-out
evaluation. It is never calibrated separately by runtime type, profile, corpus, method, or
evaluation set.

**Rejected: full-representation FiLM.** The current implementation allows lambda to rescale and
bias context, candidate, metadata, and recurrent-history coordinates. Identity initialization
protects only the initial checkpoint; after training, lambda can change semantic interpretation
rather than only privacy weight.

**Rejected for the first prototype: privacy-only FiLM.** It protects utility better than the
current implementation but gives the controller unnecessary capacity to learn profile-specific
action preferences and non-monotone behavior. The selected privacy signal is scalar, so scalar
weighting is the matching architecture.

**Rejected for the first prototype: learned per-profile slopes, profile embeddings, and separate
profile heads.** They can space operating points more flexibly but turn the finite menu into several
small policies and weaken the meaning of ordered lambda values. If the fixed controller cannot
produce useful distinct operating points, that result reopens this fork; richer conditioning is
not an automatic implementation escalation.

**Required invariants.** Tests must establish exact lambda-zero identity, lambda-invariance of
utility and privacy predictions, nonnegative `alpha`, local monotonicity of expected predicted
privacy for a fixed state and legal menu, absence of profile-identity inputs, and one globally
shared controller scale. Document-level monotonicity remains an empirical diagnostic because
sequential legal menus can change after earlier actions.

**Crux.** Lambda stops selecting a different semantic policy and becomes one transparent scalar
weight on an independently learned privacy prediction.

## Document context uses candidate-conditioned attention over a frozen bidirectional token bank

**Decision.** Replace the current candidate-independent local CLS/occurrence-mean context with a
candidate-conditioned read over the complete frozen document token bank. The base text encoder is
bidirectional and frozen. Target, relative-position, occurrence, and candidate-query projections,
attention pooling, and utility projections are trainable.

```text
doc_orig --> overlapping chunks --> frozen bidirectional encoder --> cached token states
occurrence offsets -----------------------------------------------> target/position features

source + candidate + type --> relation embedding --> action query --------+
cached token states + target/position features ---------------------------+--> trained attention
semantic decision history ------------------------------------------------+        |
                                                                                   v
                                                                    candidate-specific context
                                                                                   |
                                                                                   v
                                                                           utility tower
```

The context representation is action-specific. Two legal candidates for the same decision query
the same cached document through different source-to-candidate relation embeddings. The policy can
therefore retrieve different evidence when judging whether `solid organ transplant` versus
`medical procedure` preserves the document's task-relevant meaning.

The prototype tokenizes the complete document into 512-token encoder inputs with 64 source-token
overlap, preserving every source token in at least one chunk. It does not truncate the frozen ACI
training/evaluation documents to a single window. Encoder outputs are cached by document,
environment, model revision, tokenizer revision, and chunking configuration.

Target features mark every occurrence controlled by the current decision, distinguish other
controlled occurrences from ordinary text, and encode relative position to the nearest current
occurrence. Repeated occurrences are combined by candidate-conditioned attention, not unconditional
mean pooling. The utility representation retains a direct target-span summary alongside local and
full-document attended summaries so global retrieval cannot erase the controlled span.

**Retained baseline: local CLS/mean pooling.** The existing frozen local representation remains the
cheap control. It is not the production candidate because it is candidate-independent, discards
occurrence roles, and cannot expose most long-range QA-v2 dependencies.

**Retained baseline: bidirectional local target-span pooling.** Pooling explicitly marked target
tokens from a bidirectional local window tests whether anchoring alone resolves the problem. It
still cannot represent distant dependencies and is therefore not the selected architecture.

**Rejected: unidirectional target-span state.** A causal state at the controlled span sees only
left context. Clinical explanations and linked evidence commonly appear after the mention, so the
same semantic relation would become observable or invisible according to mention order.

**Rejected for the first prototype: full encoder fine-tuning.** The current data and reward support
do not justify training the base encoder. Full tuning removes simple frozen caches, raises cost,
and makes it difficult to distinguish a context-readout improvement from corpus memorization.
Train the attention/readout layers first. LoRA or partial unfreezing requires a new measured fork
after the frozen architecture demonstrates valid context use.

**Escalation, not initial architecture: candidate-rendered cross-encoder.** Replacing the controlled
surface with each candidate before encoding more directly represents what the remote model sees,
but multiplies encoder work by candidate count and complicates caching and sequential state. It is
considered only if candidate-query attention fails validated action-sensitive counterfactuals.

The first prototype attends over `doc_orig` plus semantic selected-action history. It does not
re-encode a partially rewritten document after every decision. If interaction diagnostics show
that the GRU/history cannot represent prior rewrites, partially rendered state becomes a separate
architecture fork rather than an implicit extension.

**Required evidence.** Compare the existing local CLS/occurrence-mean baseline, bidirectional local
target-span pooling, and the selected full-document candidate-conditioned attention under the same
frozen encoder, utility/counterfactual artifacts, parameter budget, document split, and rollout
budget. Require context-swap and candidate-swap sensitivity, gains on long-range and multi-decision
assertions, no regression on local assertions, and acceptable cache/runtime measurements.

**Crux.** Context stops being one candidate-independent local vector and becomes a candidate query
over every document region that can establish whether that specific generalization preserves
utility.

## Shared encoder checkpoint is BioClinical ModernBERT base

**Decision.** Fix both the frozen relation encoder and frozen document encoder to
[`thomas-sounack/BioClinical-ModernBERT-base`](https://huggingface.co/thomas-sounack/BioClinical-ModernBERT-base)
at immutable revision `c3648aa87af95837c809e6f0c5f85d08160db437`. It is a 150M-parameter
ModernBERT-base model with hidden size 768. The prototype uses its standard Transformers
implementation without remote model code.

The model is the best first-prototype fit because it preserves direct compatibility with the
existing ModernBERT policy path while adding broad biomedical and clinical continued pretraining.
Its 8,192-token capacity is useful future headroom, but it is not the selection argument: the
approved token-bank design still uses complete-document coverage through overlapping 512-token
inputs. Domain representation quality, standard integration, and base-size efficiency are the
load-bearing reasons.

**Evaluation contamination boundary.** BioClinical ModernBERT reports ACI-BENCH among its
pretraining sources and lists all 207 documents. The current artifact contains the 67 documents
`aci/D2N001` through `aci/D2N067`, corresponding to the ACI training corpus. ACI may be used for
prototype development and ranker training, but not to select the encoder, compare encoders, or
support an out-of-corpus generalization claim. Those conclusions require a non-ACI clinical
validation corpus not present in encoder pretraining.

### Considered models

**Selected: `thomas-sounack/BioClinical-ModernBERT-base`.** It combines clinical-note and
biomedical-literature adaptation with the same base architecture and 768-dimensional interface as
the existing `answerdotai/ModernBERT-base` path. Its immutable revision is part of the experiment
identity and cache key.

**Retained implementation baseline: `answerdotai/ModernBERT-base`.** The former default remains a
useful same-architecture, non-clinical-domain control. It is not the selected prototype encoder
because the experiment specifically targets medical documents.

**Rejected for the first prototype: `thomas-sounack/BioClinical-ModernBERT-large`.** The large
checkpoint has 396M parameters and a larger hidden representation. Its reported gains over base
are modest and mixed across the published task table, while encoding latency, memory use, and
frozen-token cache size all increase. It may be tested only after the base architecture passes the
context-observability gates and an encoder-capacity ablation is justified.

**Deferred: `Sifal/ClinicalMosaic`.** ClinicalMosaic is a credible clinical-reasoning candidate,
trained on MIMIC-IV notes and evaluated on MedNLI. It is not in the first bake-off because loading
requires trusted remote code, its model license invokes the MIMIC Data Use Agreement, and its
recommended normalization complicates exact source-offset preservation. It may be reconsidered
after code review and an offset-preserving preprocessing design.

**Rejected pending artifact repair: `Simonlee711/Clinical_ModernBERT`.** Its model card describes
an 8,192-token ModernBERT, while the published configuration declares
`MosaicBertForMaskedLM`, `model_type: bert`, and `context_length: 1024`. This mismatch creates a
material risk of loading a different architecture than the one described. It is ineligible until
an independent load, parameter-key, tokenizer, and output-parity audit resolves the discrepancy.

**Retained clinical fallback: `emilyalsentzer/Bio_ClinicalBERT`.** This standard 110M BERT was
continued from BioBERT on all MIMIC-III notes and is simple to integrate. It is not selected because
clinical continued pretraining used 128-token sequences, making it a weaker match for the
prototype's 512-token contextual chunks. It remains the low-risk clinical-domain fallback and a
useful non-ACI comparison arm.

**Rejected for the target corpus: `emilyalsentzer/Bio_Discharge_Summary_BERT`.** Its adaptation is
limited to MIMIC-III discharge summaries. That specialization does not match the current
outpatient encounter dialogues and general medical-document target. Reconsider it only for a
deployment and evaluation corpus explicitly restricted to discharge summaries.

**Retained biomedical control: PubMedBERT.**
`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` is a standard 110M BERT pretrained
from scratch on PubMed abstracts and PMC full text. Its provenance and biomedical terminology make
it a strong control, but the absence of clinical-note and dialogue pretraining makes it less likely
than the selected model to represent EHR shorthand and encounter language. It should be included
when testing whether clinical-note adaptation adds value beyond biomedical literature pretraining.

**Model artifacts.**
[`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base),
[`thomas-sounack/BioClinical-ModernBERT-base`](https://huggingface.co/thomas-sounack/BioClinical-ModernBERT-base),
[`thomas-sounack/BioClinical-ModernBERT-large`](https://huggingface.co/thomas-sounack/BioClinical-ModernBERT-large),
[`Sifal/ClinicalMosaic`](https://huggingface.co/Sifal/ClinicalMosaic),
[`Simonlee711/Clinical_ModernBERT`](https://huggingface.co/Simonlee711/Clinical_ModernBERT),
[`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT),
[`emilyalsentzer/Bio_Discharge_Summary_BERT`](https://huggingface.co/emilyalsentzer/Bio_Discharge_Summary_BERT),
and
[`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext).

**Crux.** Use the strongest simple drop-in clinical ModernBERT at base scale, but prevent its ACI
pretraining contamination from becoming evidence for encoder quality or generalization.

## Candidate features use an ordered shared pair encoder

**Decision.** Encode runtime type, canonical source, and rendered candidate jointly with the same
pinned frozen BioClinical ModernBERT used for the document token bank. Pool contextualized tokens
by field, construct a fixed ordered pair vector from type, source, candidate, signed difference,
and elementwise product, then apply separate trainable utility and privacy projections.

```text
type + source + candidate --> one frozen BioClinical ModernBERT forward
                                      |
                         field-aware contextual means
                                      |
            [type, source, candidate, candidate-source, source*candidate]
                              /                         \
                    utility projection          privacy projection
```

The candidate is never represented as a free-standing phrase in the selected architecture.
Joint encoding lets source tokens condition candidate tokens and preserves direction:
`kidney transplant -> solid organ transplant` need not equal the reversed narrowing relation.
The fixed shared pair vector has no trainable parameters, so privacy gradients cannot alter the
utility projection and utility gradients cannot alter the privacy projection.

Before full hybrid RL, run the profile-held-out fitness spike specified in the normative
architecture. Compare joint pair encoding against candidate-only and independent bi-encoder
baselines under identical frozen encoder, head budget, data, seeds, and runtime accounting. The
spike must test direction, hard negatives, grounded privacy ordering, contextual utility outcomes,
and operational cost; embedding geometry alone is not selection evidence.

### First escalation: BioLORD-2023 branch-isolated auxiliary features

[`FremyCompany/BioLORD-2023`](https://huggingface.co/FremyCompany/BioLORD-2023) is the first
candidate-feature escalation because its definition- and knowledge-graph-grounded training is more
aligned with biomedical concept hierarchy than ordinary retrieval embeddings. It is not enabled by
default. Its IHTSDO/NLM licensing and ontology coverage require review before use.

Try this path only when the shared pair encoder's errors indicate missing concept hierarchy rather
than failures elsewhere:

- legal generalizations are confused with reversals or sibling concepts;
- related but non-substitutable medical concepts collapse to the same judgment;
- grounded within-profile privacy ordering fails on held-out medical profiles;
- unseen-profile failures concentrate on ontology-rich concepts;
- local, correctly retrieved contexts still fail on category-preserving candidates.

Do not try BioLORD to address long-range context retrieval, QA-reader instability, bad count
grounding, lambda control, or arbitrary user concepts outside its ontology coverage. Add frozen
ordered BioLORD pair features to the failing branch only: a privacy-only arm for hierarchy or
privacy-order errors, or a utility-only arm for local category-preservation errors after context
retrieval is verified. Do not augment both branches in the initial ablation and do not replace the
document encoder.

### Other considered feature sources

**Rejected: legacy candidate-only BioClinical ModernBERT CLS.** The current implementation embeds
only `action.fill`. It cannot directly represent whether the candidate is a broadening, narrowing,
sibling, or unrelated concept relative to the source, and CLS is not the selected model's native
sentence-embedding objective.

**Retained diagnostic baseline: independent shared-model bi-encoder.** Encode source and candidate
separately with the pinned model, then concatenate ordered algebraic pair features. Phrase vectors
cache efficiently and preserve direction through the downstream algebra, but source and candidate
cannot condition one another inside the frozen encoder. This isolates whether joint self-attention
adds value.

**Not selected: `NeuML/bioclinical-modernbert-base-embeddings`.** This is the closest engineering
alternative: it inherits the selected clinical ModernBERT base, uses 768-dimensional mean pooling,
and integrates through Sentence Transformers. Its contrastive fine-tuning uses PubMed
title-abstract and similar-title pairs, so it optimizes literature retrieval similarity rather than
truthful source-to-generalization direction. It also inherits the base encoder's ACI contamination.

**Not selected: `NeuML/pubmedbert-base-embeddings`.** It is a mature biomedical-literature
similarity model, but lacks clinical-note adaptation and has the same retrieval-versus-abstraction
objective mismatch as the BioClinical NeuML model.

**Not selected: `abhinand/MedEmbed-small-v0.1` or `abhinand/MedEmbed-base-v0.1`.** These provide
efficient medical retrieval embeddings trained on synthetic query-positive-negative examples.
Their query-response objective is useful for retrieval but does not establish hierarchy,
substitutability, or ordered generalization. The small model is a latency baseline only.

**Rejected: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`.** SapBERT is optimized to align
synonymous biomedical entity names. The ranker must distinguish exact synonymy from broader,
narrower, sibling, and merely related candidates, so synonym collapse solves a different problem.

**Rejected: `ncbi/MedCPT-Query-Encoder` plus `ncbi/MedCPT-Article-Encoder`.** MedCPT's asymmetric
query-to-article retrieval training is valuable for literature search, but its units and objective
do not match short source-to-lattice-candidate relations.

**Escalation after BioLORD: frozen clinical NLI or pairwise cross-encoder.** A pair model can score
an explicit directional statement such as “source is a kind of candidate” and may expose stronger
entailment features. It adds another model, template sensitivity, per-pair cost, and a feature space
different from the document encoder. Consider it only if both the selected pair encoder and
BioLORD auxiliary fail validated directional cases.

**Deferred: ontology definitions and parent paths.** Candidate labels may be augmented with
artifact-grounded definitions or lattice paths when every applicable profile supplies them under a
versioned contract. They are not implicit model inputs because user-defined profiles may lack
ontology coverage and inconsistent definitions would create a new metadata shortcut.

**Rejected for the first prototype: trainable encoder adapters.** LoRA or a small candidate
transformer could adapt relation features but weakens frozen-cache semantics and increases
memorization risk on the small profile corpus. First determine whether frozen pair features are
observably insufficient.

**Model artifacts.**
[`NeuML/bioclinical-modernbert-base-embeddings`](https://huggingface.co/NeuML/bioclinical-modernbert-base-embeddings),
[`NeuML/pubmedbert-base-embeddings`](https://huggingface.co/NeuML/pubmedbert-base-embeddings),
[`abhinand/MedEmbed-small-v0.1`](https://huggingface.co/abhinand/MedEmbed-small-v0.1),
[`abhinand/MedEmbed-base-v0.1`](https://huggingface.co/abhinand/MedEmbed-base-v0.1),
[`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext),
and
[`ncbi/MedCPT-Query-Encoder`](https://huggingface.co/ncbi/MedCPT-Query-Encoder).

**Crux.** Similarity is not generalization. Start with one ordered clinical pair encoder, and add
ontology-grounded features only when held-out errors specifically demonstrate missing hierarchy.

## Action history uses selected-action cross-attention

**Decision.** Replace the legacy GRU with candidate-conditioned cross-attention over explicit prior
selected-action utility records. Treat no history and a corrected utility-only GRU as diagnostic
spike baselines. Do not use a Decision Transformer for the first prototype.

```text
prior selected actions
  [utility relation, mode, type, source position]
                         |
                         v
                set-valued memory M_<j
                         ^
                         |
current candidate --> [utility relation + retrieved document context]
                         |
                         v
            candidate-specific history h_hist(j, a)
                         |
                         v
                    utility tower
```

The ranker needs selective access to earlier semantic choices, not an opaque summary of sampling
order. A current candidate can directly retrieve the prior actions relevant to its utility while
ignoring unrelated ones. Memory excludes count, predicted privacy, lambda, authored index, menu
size, profile identity, and QA routing identifiers. It uses original source positions but no
selection-step positional embedding, so permuting equivalent memory rows does not change the
policy output.

This choice matches the environment's semantics. The final rewrite is assembled from the complete
legal action vector; semantic substitutions are not intrinsically ordered. Sequential sampling
order matters only to external legality state such as fill collisions. The ACI corpus currently
has 4--38 decisions per document (median 11), making one cross-attention block operationally cheap.
The prototype nevertheless keeps the deterministic first-occurrence walk. Reverse and seeded
alternative orders are diagnostics only; material sensitivity triggers a two-pass
draft-and-refine fork rather than silent order randomization.

### Retained baseline: no history

Score each candidate from the original document, decision features, and candidate relation only.
This is the simplest, fastest, and most parallel architecture, and it avoids inventing interaction
state when most decisions may be conditionally independent. Its failure mode is policy-state
aliasing when utility depends on combinations of selected actions. If selected-action attention
does not improve validated multi-decision outcomes over this baseline, no history becomes the
preferred production choice.

### Retained diagnostic baseline: corrected utility-only GRU

The corrected GRU receives the same count-blind semantic records admitted to selected-action
memory. It does not receive the legacy action vector containing normalized count, authored level
position, menu size, lambda, or profile identity.

```text
selected utility record --> GRU(prefix state) --> one prefix summary
current candidate + prefix summary ------------> utility tower
```

This arm is cheap and linear in decision count, and it tests whether any causal prefix summary is
enough. It is not the production default because it imposes arbitrary sampling-order sensitivity,
compresses all prior actions through a fixed bottleneck, tends toward recency bias, and gives every
candidate the same history summary. It is retained to diagnose whether cross-attention gains come
from explicit retrieval rather than merely adding history capacity. Failure of the selected module
does not automatically promote this GRU.

### Rejected: Decision Transformer

A Decision Transformer would model an ordered trajectory conditioned on a desired return:

```text
desired return + states + actions --> causal Transformer --> next action
```

That is a different training formulation, not a drop-in history encoder. It requires a suitable
offline trajectory corpus and return-conditioning contract, preserves dependence on an arbitrary
decision order, increases model and serving complexity, and weakens the current separation between
semantic utility prediction and explicit lambda-controlled privacy preference. The ranker already
has a small discrete action menu, direct online or cached reward evaluation, and short documents;
it does not need sequence-model capacity to retrieve one relevant prior choice. Reconsider this
only if large offline trajectory data becomes the primary training regime and ordered long-horizon
planning is empirically load-bearing.

**Crux.** Store selected semantic actions explicitly and let each candidate retrieve what matters.
If that does not beat no history on real multi-decision utility, use no history rather than a more
opaque sequence model.

## Privacy pretraining uses a fixed-sigma standardized mean

**Decision.** Replace the primary heteroscedastic log-count head with a deterministic mean head on
train-standardized log counts. Normalize the type, source, candidate, and Hadamard relation blocks
with train-only statistics, remove the redundant candidate-minus-source block, and begin with one
linear output. Estimate a single post-hoc dev-residual scale clamped to `[0.3, 3.0]`; learned sigma
is an ablation only.

Train complete profiles with equal profile weight using standardized SmoothL1 plus profile-relative
Huber at weight `1.0`. Aggregate decisions before profiles so training and evaluation weight menus
identically. Ranking is off by default. Its registered ablation uses weight `0.05` and bounded
regression of pair differences, not sign-only logistic separation. Neural diagnostic arms use a
fixed non-trainable projection to one shared trainable head and report their native parameter
counts; repeated-feature parameter matching is rejected.

**Evidence.** The previous head reached converged within-menu ordering while reporting held-out NLL
`17.6`, consistent with variance overconfidence rather than absent mean signal. Its promotion gate
was impossible because semantic ordering had to strictly exceed an authored-position oracle at
`1.0`. The advertised matched comparison also used approximately `1.05M` semantic parameters
against approximately `263K` candidate-only parameters.

**Gate consequence.** Profile-relative calibration and within-menu ordering are the primary
controller metrics; selected-action regret and lexical/semantic counterexamples are also blocking.
Candidate-only is the competitive baseline. Authored position is an oracle-ceiling reference with
a non-inferiority margin, while mode/type-only and train-profile mean remain sanity floors. NLL,
coverage, and fixed sigma are report-only for policy fitness. Treating predicted counts or
intervals as audit claims requires a separate distributional-audit gate.

Promotion uses paired per-profile deltas aggregated across at least three preregistered seeds. A
95% bootstrap interval must support improvement on one primary metric while the other remains
inside its preregistered non-inferiority margin. Calibration and selected-regret margins are
distinct because the former measures a continuous controller-score error while the latter is a
direct action-selection consequence. Predicted log-count differences within `1e-6` are ties.

The incompatible artifact schema is versioned
`ranker-v2-semantic-privacy-v2`/`ranker-v2-semantic-privacy-metrics-v2`/
`ranker-v2-semantic-privacy-diagnostic-v2`. Its checkpoint binds the run protocol, seed count,
metric and diagnostic hashes, counterexample-set hash, and promotion verdict. Policy admission is
fail-closed on all non-promotion and count-basis checkpoints; an explicit development override
relaxes only the promotion-evidence requirement.

**Rejected: retain blocking NLL.** Sigma is not consumed by the controller, so variance calibration
cannot veto a controller whose held-out profile-relative means are fit for purpose. It remains
visible and can block distributional claims separately.

**Rejected: unbounded logistic rank loss.** A sign-only objective keeps increasing unequal
separation after the correct ordering is achieved and can distort calibrated log-count
differences. Bounded difference regression has a finite target.

**Rejected: learned heteroscedastic sigma at this data volume.** The available profile count is too
small to distinguish transferable aleatoric structure from train-profile confidence. Post-hoc
residual scale is the narrower prototype; grouped or conformal uncertainty remains future audit
work.

**Crux.** Fit the controller input directly with a small profile-balanced mean head, and quarantine
uncertainty claims behind their own audit gate.

## Additive controller is superseded by a lexicographic residual actor

**Date:** 2026-08-05

**Decision.** Retain the validated frozen semantic representation stack, candidate-conditioned
context readout, and selected-action cross-attention memory. Train that stack as a utility-only
policy, then freeze it. Serve the strict positive regime through a zero-initialized
action-conditioned residual actor trained by separate utility and count policy gradients under a
training-only document utility dual. The deployed policy performs an ordinary neural forward pass;
it does not enumerate or filter an action slate.

The selected residual architecture is `lexicographic-residual-ln-gelu16-v1`:

```text
non-affine LayerNorm over the 4608-dimensional semantic action feature
-> Linear(4608, 16, bias=True)
-> non-affine LayerNorm(16)
-> GELU
-> Linear(16, 1, bias=False), zero initialized
```

The input and pre-activation norms address the measured saturation of the former tanh gain head.
Width `16` is the smallest nonlinear head that passed the architecture preflight; width `32` added
no material differentiation and is not the default.

**Gradient ownership.** Count and positive-regime utility advantages update the residual actor.
Count cannot update the frozen utility base, utility feature extractor, selected-action memory, or
utility critic. Utility cannot update the count critic. Critics consume detached frozen semantic
features and share no trainable parameters. Per-document dual values are projected nonnegative
optimizer state, not actor input and not deployment calibration.

**Why this supersedes the selected semantic privacy head.** The semantic privacy-head program
tested whether candidate wording could predict count. Its transfer gate failed, and production
experiments consequently used exact direct-count targets. More importantly, the additive
controller compared count pressure against arbitrary policy-logit margins. Repeated alpha,
gap-scaling, softcap, gain, hinge, projection, KL, and sensitivity interventions changed authority
without repairing that solution-concept mismatch. The new design keeps exact counts as rewards and
lets both objective gradients reach the behavior-producing residual actor.

**Rejected: deterministic lexicographic selector.** Value-based lexicographic filtering is
literature-backed, but it requires an accurate utility value model and an inference-time action-set
operator. The user rejected deterministic selection for deployment, and the standardized four-vector
gate was support-limited. `src/cloak/ranker/lexicographic.py` remains an offline oracle/test utility
only.

**Rejected: retain additive alpha with a stronger or state-conditioned gain.** The measured gain
head collapsed to a uniform field, and after its saturation defect was identified the repaired
authority interval still depended on arbitrary utility-logit scale. Increasing controller capacity
does not change the weighted-sum solution concept.

**Retained alternative: constrained policy optimization.** If the first-order primal-dual update
violates utility despite valid advantages, escalate to a CPO-style trust-region constraint rather
than reviving additive control. If product semantics change from strict utility priority to a smooth
Pareto tradeoff, MO-MPO/LP3 becomes the relevant alternative and requires a separate fork.

**Evidence and literature.** The full classification and comparison are in
[Trainable multi-objective RL for ranker v2](../../research/ranker-v2-trainable-multi-objective-rl-review.md).
The selected training family follows [Lexicographic Multi-Objective Reinforcement Learning](../../../research-wiki/papers/skalse2022_lexicographic_morl.md)
([arXiv 2212.13769](https://arxiv.org/abs/2212.13769)) and multi-timescale constrained updates from
[Reward Constrained Policy Optimization](../../../research-wiki/papers/tessler2018_reward_constrained_policy.md)
([arXiv 1805.11074](https://arxiv.org/abs/1805.11074)). The trust-region escalation is
[Constrained Policy Optimization](../../../research-wiki/papers/achiam2017_constrained_policy_optimization.md)
([arXiv 1705.10528](https://arxiv.org/abs/1705.10528)). The smooth-tradeoff alternative is
[MO-MPO](../../../research-wiki/papers/abdolmaleki2020_distributional_view_multiobjective.md)
([arXiv 2005.07513](https://arxiv.org/abs/2005.07513)) with preference-selection precedent from
[LP3](../../../research-wiki/papers/huang2022_constrained_multiobjective_reinforcement.md)
([PMLR](https://proceedings.mlr.press/v164/huang22a.html)).

**Scope.** This decision authorizes one cache-only four-document mechanism experiment. It does not
authorize full-corpus training, held-out generalization claims, or privacy claims.

**Crux.** Freeze what represents utility; train one semantic residual policy with both behavioral
gradients; use utility violation to govern the update rather than adding count to arbitrary logits.

## Frozen-feature residual is superseded by three-tier side adaptation

**Date:** 2026-08-05

**Supersedes:** the representation-freeze portion of “Additive controller is superseded by a
lexicographic residual actor.” The primal-dual actor objective, separate critics, exact count
rewards, and immutable utility reference remain selected.

**Trigger.** A head-only residual over `stop_gradient(x_U)` protects the utility reference, but it
also assumes that the utility branch's final 4,608-dimensional feature is a sufficient statistic
for learning the secondary objective on unseen states. BC, verified ExIt, and utility-only RL did
not train that representation to preserve every semantic distinction useful for count-sensitive
generalization. Freezing the complete semantic stack therefore turns utility initialization into
an irreversible information bottleneck. Unfreezing the shared stack would solve the bottleneck by
destroying the immutable utility reference and allowing count gradients to redefine utility.

**Decision.** Use a three-tier freeze policy:

1. **Tier 1 — frozen clinical substrate.** The pinned BioClinical ModernBERT encoder, tokenizer,
   token/relation banks, masks, frozen action metadata, and canonical selected-action records are
   frozen from the start.
2. **Tier 2 — utility semantic branch.** Candidate relation, context readout, selected-action
   memory, interaction features, and utility head are trained by BC, verified ExIt, and
   utility-only structured RL, then frozen and hashed.
3. **Tier 3 — private lexicographic semantic side path.** Frozen Tier-2 maps receive private
   rank-4 low-rank deltas on the relation, context-attention, context-projection, and
   selected-action-memory transformations. A normalized GELU-16 head maps the resulting
   `x_lex` feature to the residual logit. Utility and count actor surrogates update Tier 3 only.

The side path consumes the same detached Tier-1 substrate and canonical sequential state as the
utility branch. It does not consume only `x_U`, and it does not receive count values, authored
positions, lambda, or dual values. Frozen utility mode/type embeddings may be reused as semantic
metadata.

**Initialization.** Every low-rank delta uses zero-product initialization: one factor is randomly
initialized and its output factor is exactly zero. The residual head is
`LayerNorm(4608, affine=False) -> Linear(4608,16,bias=True) -> LayerNorm(16,
affine=False) -> GELU -> Linear(16,1,bias=False)`, with the final map exactly zero. The served
lexicographic logits therefore equal the frozen utility logits before the first update while all
selected Tier-3 routes remain trainable.

**Gradient ownership.** Utility and count actor surrogates traverse the residual head and private
semantic deltas in reverse mode. They stop at the frozen maps and detached Tier-1 substrate. The
utility critic consumes detached `x_U`; the count critic consumes detached `x_lex`; critic losses
do not update either actor branch. Document dual loss updates only `mu_d`. Counts remain rewards,
never features.

**Served contract.** The prototype serves only `z_lex = u_theta + r_psi`. The frozen utility path
remains callable for utility-reference construction, parity checks, ablations, and audit, but is
not exposed as a user-selectable report mode. This removes a product distinction that was useful
for experimentation but irrelevant to the target report.

**Rejected: fully frozen semantic stack plus residual head.** It is maximally auditable and cheap,
but forces secondary behavior through a representation trained only for the primary objective. It
remains a diagnostic ablation, not the selected actor.

**Rejected: unfreeze the shared utility branch.** This permits a single expressive actor, but count
and lexicographic utility gradients can alter `u_theta`, invalidate `b_d`, and erase the exact
utility-reference identity that gives the constraint operational meaning.

**Rejected: encoder LoRA or partial encoder fine-tuning.** The current failure does not establish
that the clinical token substrate is deficient. Encoder adaptation increases memory, invalidates
frozen token caches, and creates the broadest interference route. Reconsider only after the
post-encoder side path fails a representation-use gate.

**Rejected: duplicate the complete semantic tower.** A full second tower is expressive but wastes
capacity and weakens the controlled comparison. Rank-4 deltas provide an explicit, auditable
adaptation budget on the transformations most directly responsible for candidate/context/history
semantics.

**Cost and auditability.** The architecture figure's roughly 227K trainable Tier-3 label is a
planning estimate, not an artifact contract. The architecture manifest records the derived exact
count from the resolved logical adapter maps and fails closed on target-map drift. The design adds
a second post-encoder semantic pass while preserving the expensive frozen encoder cache.

**Evidence and companions.** See the normative
[architecture specification](ranker-v2-architecture.md), the
[interactive RL specification](interactive-ranker-v2.md), the
[implementation plan](../../plans/2026-08-05-ranker-v2-lexicographic-actor-critic.md), the
[architecture report](../../html/interactive-ranker-v2.html), and the
[trainable multi-objective RL review](../../research/ranker-v2-trainable-multi-objective-rl-review.md).
The optimization family remains grounded in
[Lexicographic Multi-Objective Reinforcement Learning](../../../research-wiki/papers/skalse2022_lexicographic_morl.md)
([arXiv 2212.13769](https://arxiv.org/abs/2212.13769)),
[Reward Constrained Policy Optimization](../../../research-wiki/papers/tessler2018_reward_constrained_policy.md)
([arXiv 1805.11074](https://arxiv.org/abs/1805.11074)), and
[Constrained Policy Optimization](../../../research-wiki/papers/achiam2017_constrained_policy_optimization.md)
([arXiv 1705.10528](https://arxiv.org/abs/1705.10528)).

**Crux.** Freeze the utility function, not every semantic transformation available to the
lexicographic actor.
