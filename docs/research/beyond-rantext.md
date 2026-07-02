---
type: research
status: current
created: 2026-07-01
updated: 2026-07-01
tags: [rantext, inferdpt, privacy, research-directions, substitution, reconstruction, pii, budgeting]
companion: [rantext-limitations.md, rantext.md, embedding-map.md]
---

# Beyond RANTEXT — research directions

Given the failure taxonomy in [`rantext-limitations.md`](rantext-limitations.md), this report maps the
design space *past* per-token metric-LDP token replacement. Six directions (RD0–RD5), each with the
failure it targets, then a **Findings** list — one entry per approach/paper/system, each stating what
it does and *why it fits this direction* — followed by that direction's cross-cutting connections,
conclusions, and open problems. Registered papers link to their `research-wiki/papers/` page; canonical
references are cited inline with arXiv/DOI/standard IDs.

## Definitions

- **Protectable unit:** the atom the policy operates on — token, subword, word, or normalized *entity
  span*. RANTEXT's unit is the cl100k subword surface; RD0 argues for the span.
- **Quasi-identifier (QI):** an attribute non-identifying alone but identifying in combination (gender,
  ZIP, birth date). **Direct identifier:** identifying alone. The line is context-dependent.
- **Budget-allocation policy π:** `span → (mechanism, local ε)`, replacing scalar ε. **Role:**
  identifier / format-bearing / value-bearing / premise-bearing / filler. **Salience:** measured
  downstream influence.
- **Surrogate:** a type/role-consistent replacement (typed placeholder, pseudonym, format-preserving
  ciphertext, bucketed value) chosen by rule, not embedding distance.
- **Reversible transform:** client-keyed, invertible substitution — de-sanitizing the response is a
  deterministic reverse-map, not a learned guess.
- **Post-processing immunity:** any function of a DP output is DP with the same ε — a *local* stage
  after the private channel costs zero budget (Dwork & Roth 2014).
- Failure labels **F1a/F1b/F2/F3a/F3b/F4**, **E1/E2/E3**, **A1/A2/A3** are defined in
  [`rantext-limitations.md`](rantext-limitations.md).

## Why these six, and where they attach

The taxonomy's unifying flaw: **one scalar ε governs three quantities that must move independently**
(identity-removal, premise-preservation, extraction-signal), and improving φ is the wrong layer because
the extractor is load-bearing (E1). These directions decompose the knob and relocate the work.

| RD | Name | Kills |
|----|------|-------|
| **RD0** | Tokenizer-robust protectable units | F3a, F3b (unblocks all) |
| **RD1** | Role- & salience-aware budgeting | F1b; arbitrates F2-vs-premise |
| **RD2** | PII detection & sensitivity typing | supplies RD1/RD3/RD4; confronts A1/F4 |
| **RD3** | Structured surrogate substitution (reversible) | F2, part of F1a; shrinks E1 |
| **RD4** | Learned context-aware substitution | F1a (+F1b if joint) |
| **RD5** | Targeted reconstruction architectures | E1, E2 |

```
RD0 (units) ──┬──▶ RD2 (PII typing) ───┬──▶ RD3 (structured surrogate) ──┐
              ├──▶ RD1 (budgeting) ◀───┘                                 ├──▶ RD5 (reconstruction)
              └──────────────────────────▶ RD4 (learned substitution) ───┘
```
RD0 is foundational. RD1↔RD2 are coupled: typing feeds budgeting, and the QI/combination problem RD2
must solve *is* the A1/F4 correlation problem. RD3/RD4 are the programmable/learned substitution
branches; RD5 consumes whichever substrate they produce.

---

## RD0 — Tokenizer-robust protectable units

**Framing.** Fix the unit of protection at the normalized entity/span level — casefold, normalize, and
resolve entities *before* BPE — so casing/whitespace variants (F3a) don't create inconsistent or missed
protection, and rare proper nouns aren't silently dropped by vocabulary truncation. Replace "discard
OOV" (F3b) with typed placeholders: holes both hurt utility *and* mark where the sensitive spans were.

**Fixes:** F3a, F3b. Unblocks RD1/RD2 (stable spans to label/score), RD3/RD4 (stable spans to substitute).

**Findings**
- **CANINE** (Clark et al. 2021, [2103.06874](https://arxiv.org/abs/2103.06874)) — character-level
  encoder with convolutional downsampling, no fixed subword vocabulary. *Fit:* removes the exact-match
  lookup that causes F3a/F3b entirely — a span's identity can't be missed because there is no `V` to
  miss it in.
- **ByT5** (Xue et al. 2021, TACL, [2105.13626](https://arxiv.org/abs/2105.13626)) — UTF-8 byte-to-byte
  seq2seq, empirically more robust to noise and spelling. *Fit:* lets the *whole* pipeline (including
  the RD5 reconstructor) run vocabulary-free, so there is no OOV to drop and casing is just bytes.
- **Charformer / GBST** (Tay et al. 2022, ICLR, [2106.12672](https://arxiv.org/abs/2106.12672)) —
  learned, differentiable subword segmentation. *Fit:* keeps subword efficiency but makes the split
  soft, so a rare name isn't forced into one brittle segmentation that a fixed BPE would mis-handle.
- **Subword regularization** (Kudo 2018, [1804.10959](https://arxiv.org/abs/1804.10959)) — train over
  sampled segmentations of each word. *Fit:* if you must keep BPE, this is the cheapest hardening
  against the rare-token char-splits that make F3b's behavior unpredictable.
- **Byte-sized NER** ([1809.08386](https://arxiv.org/abs/1809.08386)) — entity tagging at the byte level.
  *Fit:* detects PII spans without a vocabulary, so the rare proper nouns most likely to *be* PII are
  not the ones the detector misses — the RD0→RD2 hand-off.
- **Microsoft Presidio** (open-source analyzer + anonymizer) — normalize → NER + pattern + context →
  typed placeholder. *Fit:* the production reference for "normalize before typing" and for replacing
  deletion with typed placeholders (`<PERSON_1>`).
- **[Hide-and-Seek (HaS)](../../research-wiki/papers/chen2023_hide_seek_has.md)** ([arXiv 2309.03057](https://arxiv.org/abs/2309.03057)) (Chen et al. 2023) —
  anonymize→de-anonymize round-trip for LLM prompts. *Fit:* demonstrates typed placeholders are stable
  enough to be reversed, validating the placeholder as a first-class protectable unit.
- **[Lison et al. 2021 SoK](../../research-wiki/papers/lison2021_anonymisation_models_text.md)** ([ACL 2021](https://aclanthology.org/2021.acl-long.323/)) —
  argues anonymisation should model spans and disclosure risk, not tokens. *Fit:* the field-level
  statement of RD0's core move (unit = span).

**Cross-cutting connections.** RD0 is upstream of everything: RD1/RD2 can only label and score stable
spans, and RD3/RD4 can only substitute them consistently if the same entity has one canonical form
across the document. It also directly interlocks with RD2's attack finding — an infilling attacker
recovers deletion holes, so RD0's "abstract, don't delete" is enforced by RD5-class adversaries.

**Conclusions.** The cheapest correct design normalizes and resolves entities to canonical spans, then
either (a) runs a byte/char model to sidestep the vocabulary entirely, or (b) keeps BPE but pins spans
with subword-robust NER and typed placeholders. Deletion is never acceptable — it is both a hole and a
marker.

**Open problems.** Tokenizer-agnostic span alignment when the *remote* model re-tokenizes placeholder
text differently than the client; canonicalization for entities with no clean surface form (nicknames,
misspellings, cross-lingual variants).

---

## RD1 — Role- and salience-aware budgeting

**Framing.** Replace scalar ε with a policy `π: span → (mechanism, local ε)` keyed on **role**
(identifier / format-bearing / value-bearing / premise-bearing / filler) and measured **salience**
(downstream influence). Scrub identifier spans hard; leave premise-bearing spans nearly intact.

**Fixes:** F1b; arbitrates the F2-vs-premise tension. Depends on RD0 (spans) and RD2 (labels).

**Findings**
- **Concentrated DP / zCDP** (Dwork & Rothblum 2016, [1603.01887](https://arxiv.org/abs/1603.01887)) —
  tighter composition of many small privacy losses. *Fit:* a per-span policy composes hundreds of tiny
  per-span ε's; zCDP is what keeps that composition from blowing the global budget.
- **Personalized / heterogeneous DP** (Jorgensen et al. 2015, ICDE, "Personalized Differential
  Privacy") — different records get different ε. *Fit:* the formal precedent that ε need not be uniform;
  RD1 is the per-span analogue for text.
- **[Mahalanobis mDP](../../research-wiki/papers/xu2020_differentially_private_text.md)** ([arXiv 2010.11947](https://arxiv.org/abs/2010.11947)) (Xu et al.
  2020) — noise shaped by local word density. *Fit:* a geometric ancestor of role-weighting — already
  makes perturbation non-uniform across the embedding space, though keyed on density, not role.
- **DynText** (ACL Findings 2025, [PDF](https://aclanthology.org/2025.findings-acl.1038.pdf)) — DP
  semantic-density sizes each token's adjacency list dynamically. *Fit:* the closest existing thing to
  RD1 — it already varies the mechanism per token; RD1 generalizes the key from density to role+salience.
- **[CusText](../../research-wiki/papers/chen2022_customized_text_sanitization.md)** ([arXiv 2207.01193](https://arxiv.org/abs/2207.01193)) (Chen et al. 2022)
  — per-token customized output sets. *Fit:* the mechanism-level hook for "different tokens, different
  treatment" — the substrate a role policy drives.
- **Adaptive Token-Weighted DP** ([2509.23246](https://arxiv.org/abs/2509.23246)) — allocate budget by
  perplexity-gauged token rarity. *Fit:* literal per-token budget allocation; validates RD1's premise
  but keys only on rarity — RD1 argues role+salience subsume it.
- **Attention rollout** (Abnar & Zuidema 2020, [2005.00928](https://arxiv.org/abs/2005.00928)) —
  aggregate attention across layers into a token-influence score. *Fit:* a cheap salience signal
  computable on the *local* model to rank premise-bearing vs filler spans.
- **Influence functions** (Koh & Liang 2017, [1703.04730](https://arxiv.org/abs/1703.04730)) — measure
  an input's effect on an output via Hessian-vector products. *Fit:* a principled "downstream influence"
  measure; the rigorous end of the salience axis (leave-one-out on the local continuation is the cheap end).
- **"Attention is not Explanation"** (Jain & Wallace 2019, [1902.10186](https://arxiv.org/abs/1902.10186))
  — raw attention weights don't reliably indicate importance. *Fit:* the guardrail — key the budget on
  attention *flow*/occlusion/gradients, never bare weights; the negative result that disciplines RD1.

**Cross-cutting connections.** RD1 is meaningless without RD2's labels and RD0's spans; conversely it is
the arbiter that lets RD3 and RD4 coexist (route identifiers to reversible surrogates, premise spans to
contextual swaps). Salience measurement borrows wholesale from interpretability — this is the clearest
case of the report's "answers live outside privacy lit."

**Conclusions.** The scalar ε is the taxonomy's single point of failure; RD1 is the direct remedy, and
the mechanism substrate (DynText, CusText) and the salience tooling (rollout, influence, occlusion)
both already exist. The missing piece is the *policy* mapping (role, salience) → (mechanism, ε), not the
components.

**Open problems.** Salience must be computed without leaking the local continuation. A content-aware
budget is itself a side-channel — knowing "this span got more noise" leaks that it was sensitive (the
DP-faithfulness corner of the taxonomy's triangle); quantifying and bounding that meta-leak is open.

---

## RD2 — PII detection & sensitivity typing

**Framing.** Detect and type sensitive content for budgeting (RD1) and substitution (RD3/RD4). Detection
error sets the whole frontier: **recall is the privacy ceiling, precision is the utility ceiling.** The
hard part is not spans but **combinations** — this is the same correlation problem as A1 and the
aggregate leakage of F4.

**Fixes:** supplies RD1/RD3/RD4; confronts A1, F4. Depends on RD0 (must run on normalized units).

**Findings**
- **k-anonymity** (Sweeney 2002, [DOI 10.1142/S0218488502001648](https://doi.org/10.1142/S0218488502001648))
  — formalizes quasi-identifiers and indistinguishability among k records. *Fit:* defines the object
  RD2 must detect (QIs) and names why surface span-detection is insufficient.
- **"Simple Demographics Often Identify People Uniquely"** (Sweeney 2000, CMU) & **Golle 2006**
  ([WPES DOI 10.1145/1179601.1179615](https://doi.org/10.1145/1179601.1179615)) — {ZIP, gender, DOB}
  re-identifies **87%** (revised to **63%** on 2000-census). *Fit:* the quantitative proof that three
  innocuous fields, each safe alone, re-identify most people — RD2's reason to exist.
- **l-diversity** (Machanavajjhala et al. 2007, ACM TKDD) — defends against homogeneity within a QI
  group. *Fit:* shows even k-anonymized data leaks through attribute *combination*; the same lesson
  transfers to text spans.
- **NISTIR 8053** (Garfinkel 2015, NIST) — de-identification taxonomy, direct vs quasi identifiers.
  *Fit:* the authoritative typing schema for RD2's role labels; standards-grade definitions.
- **[Lison et al. 2021 SoK](../../research-wiki/papers/lison2021_anonymisation_models_text.md)** ([ACL 2021](https://aclanthology.org/2021.acl-long.323/)) —
  argues past sequence labelling toward explicit disclosure-risk modelling. *Fit:* RD2's mandate stated
  at field level — detection must model inference risk, not just tag spans.
- **Text Anonymization Benchmark (TAB)** (Pilán et al. 2022, [2202.00443](https://arxiv.org/abs/2202.00443))
  — annotated corpus distinguishing direct vs quasi identifiers. *Fit:* the evaluation substrate that
  scores whether a detector catches QIs, not just direct IDs.
- **Presidio** (Microsoft, open-source) — pattern + NER + context PII detection with confidence knobs.
  *Fit:* the production baseline, and the precision/recall dial that literally sets RD2's frontier.
- **LLMs are Advanced Anonymizers** (Staab et al. 2024, [2402.13846](https://arxiv.org/abs/2402.13846))
  — LLMs infer personal attributes by reasoning over a document. *Fit:* the only demonstrated way to
  *see* QIs is an LLM that reasons about combinations — and that same LLM is the attacker (duality).
- **Re-identification by autoregressive infilling** ([2505.12859](https://arxiv.org/abs/2505.12859)) —
  recovers redacted spans from surrounding context. *Fit:* proves detect-and-delete fails (F3b holes are
  recoverable); recall must be paired with *abstraction*, not deletion — the RD0 requirement.
- **Stronger re-id via reasoning + aggregation** ([2510.09184](https://arxiv.org/abs/2510.09184)) —
  aggregates weak signals across the whole document. *Fit:* the empirical face of A1 — the attacker
  aggregates, so the detector must reason at document scale too.

**Cross-cutting connections.** RD2 is the hinge: it feeds RD1's budget and RD3/RD4's substitution, and
its hard case (combinations) is identical to the taxonomy's A1/F4. Its best-known solution (an LLM
reasoner) *is* the reconstruction adversary of RD4/RD5 — detection and attack share a model.

**Conclusions.** Span-level NER is necessary but provably insufficient; the frontier is document-level
QI reasoning, which today means an LLM detector. Because that detector doubles as the attacker, RD2 must
be evaluated adversarially (TAB + a re-identification attacker), never by span-F1 alone.

**Open problems.** Document-level, cross-span QI detection is essentially undone — the concrete form of
"defeat A1." Running an LLM reasoner *locally* without it becoming a new leakage surface; calibrating
recall to a target realized privacy rather than a span-F1 number.

---

## RD3 — Structured surrogate substitution (programmable, reversible)

**Framing.** Replace a sensitive span with a surrogate chosen by **type and role**, not embedding
distance; map identical entities identically. Where the transform is **client-keyed and invertible**
(FPE for structured values; a client-held table for named entities), de-sanitizing the response is a
deterministic reverse-map — no learned reconstruction, no plaintext to the remote model.

**Fixes:** F2, part of F1a. Depends on RD1 (role → mechanism). Feeds RD5 — the reversible fraction needs
no reconstruction, shrinking the extractor's job to the non-reversible residue.

**Findings**
- **NIST SP 800-38G — FF1 / FF3-1** ([NIST](https://csrc.nist.gov/pubs/sp/800/38/g/final)) — standardized
  format-preserving encryption: encrypt a value to the same format, keyed and invertible. *Fit:* for
  structured value tokens (SSN, card, date) the client key makes de-sanitize a deterministic *decrypt* —
  zero reconstruction cost, zero leak, exact reversibility.
- **FFX mode** (Bellare, Rogaway, Spies 2010) — Feistel construction underlying FPE. *Fit:* establishes
  that "format-preserving + reversible" is a solved crypto primitive, not something to learn.
- **User-level metric-DP via Earth-Mover distance** (Imola et al. 2024,
  [2405.02665](https://arxiv.org/abs/2405.02665)) — mDP capturing magnitude and spatial change of values.
  *Fit:* for numeric spans you'd rather coarsen than encrypt, gives a principled *bucketed-range*
  mechanism with a guarantee — the value-bearing-role branch of RD3.
- **Hiding in Plain Sight (HIPS)** (Carrell et al. 2013, JAMIA,
  [DOI 10.1136/amiajnl-2012-001034](https://doi.org/10.1136/amiajnl-2012-001034)) — replace PHI with
  *realistic* fake surrogates. *Fit:* realistic surrogates defeat the "deletion marks the span" leak
  (F3b) and keep text natural enough that the remote LLM reasons correctly over it.
- **BRATsynthetic** ([2210.16125](https://arxiv.org/abs/2210.16125)) — compares consistent / random /
  Markov surrogate strategies; consistent mapping roughly halves residual leakage. *Fit:* quantifies why
  RD3 needs *consistent* entity→surrogate mapping (coreference-stable), not per-occurrence randomness.
- **[Hide-and-Seek (HaS)](../../research-wiki/papers/chen2023_hide_seek_has.md)** ([arXiv 2309.03057](https://arxiv.org/abs/2309.03057)) (Chen et al. 2023) —
  H masks entities, S trains a local model to de-anonymize the LLM output. *Fit:* the reversible
  round-trip realized for LLM prompts — the RD3→RD5 bridge with a client-side de-anonymizer.
- **EmojiPrompt** ([2402.05868](https://arxiv.org/abs/2402.05868)) — map sensitive spans to a symbolic
  alphabet, reversibly. *Fit:* shows the remote LLM can still operate over *non-natural* surrogates,
  widening the design space beyond plausible pseudonyms.
- **Property-preserving encryption survey** ([2312.12075](https://arxiv.org/abs/2312.12075)) —
  OPE/ORE/FPE/searchable taxonomy and what each leaks. *Fit:* situates FPE among alternatives and makes
  the leakage tradeoffs explicit for choosing a per-role mechanism.

**Cross-cutting connections.** RD3 is only as good as RD1's role labels and RD2's typing (wrong type →
wrong mechanism). Its consistency requirement needs RD0's canonical spans + coreference. Its biggest
payoff is on RD5: every reversibly-handled span is one RD5 never reconstructs and the duality can never
leak.

**Conclusions.** For identifier and format-bearing spans, RD3 dominates: crypto gives exact, zero-cost,
zero-leak reversibility that no learned method can match. It should carry the identifier/format load;
RD4 carries the rest. The engineering risk is not the cipher but consistent coreference-stable mapping
and whether the remote model reasons correctly over surrogates.

**Open problems.** Coreference-robust consistent pseudonymization across a long document; ensuring the
remote LLM treats `<PERSON_1>`/FPE ciphertext with correct type semantics; composing FPE (no DP
guarantee) with mDP value-bucketing under one accounting story.

---

## RD4 — Learned context-aware substitution

**Framing.** Replace the rule engine with a **model that reads the whole context through attention** and
emits a substitution conditioned on it — so the same surface form is handled differently by task and
sense ("bank" finance vs river; a disease name as premise vs incidental). Targets F1a directly and, if
the model jointly scores importance, F1b. **Design intent: finetune existing pretrained MLM/infilling/
seq2seq models — substitution *is* their pretraining objective.**

**Fixes:** F1a (+F1b if joint). Depends on RD0, RD1 (a learned model can subsume much of RD1).

**Findings**
- **BERT-based lexical substitution** (Zhou et al. 2019, [ACL P19-1328](https://aclanthology.org/P19-1328/))
  — partial-mask the target word, rank contextual substitutes from the MLM, no external lexicon. *Fit:*
  this *is* RD4's mechanism off-the-shelf — a contextual, sense-aware swap from a pretrained MLM.
- **ILM — Infilling LM** (Donahue et al. 2020, [2005.05339](https://arxiv.org/abs/2005.05339)) — fill an
  arbitrary span conditioned on both sides. *Fit:* "substitute a span given context" is exactly the
  infilling objective; a direct finetune target for RD4.
- **T5 span-corruption** (Raffel et al. 2020, [1910.10683](https://arxiv.org/abs/1910.10683)) & **GLM
  blank infilling** ([2103.10360](https://arxiv.org/abs/2103.10360)) — pretraining *is* masked-span
  reconstruction. *Fit:* the evidence base for RD4's "don't train from scratch" — the base models
  already perform span substitution.
- **Swords benchmark** (Lee et al. 2021, [2106.04102](https://arxiv.org/abs/2106.04102)) — high-coverage
  lexical-substitution evaluation. *Fit:* the harness to measure whether a learned substitutor picks
  meaning-preserving, sense-correct swaps — RD4's utility metric.
- **[DP-Prompt](../../research-wiki/papers/utpala2023_locally_differentially_private.md)** ([arXiv 2310.16111](https://arxiv.org/abs/2310.16111)) (Utpala et al.
  2023) — zero-shot paraphrase under an exponential mechanism on outputs. *Fit:* the whole-sequence
  *paraphrase* branch of RD4 with a DP guarantee, no finetuning.
- **[DP-BART](../../research-wiki/papers/igamberdiev2023_dp_bart.md)** ([arXiv 2302.07636](https://arxiv.org/abs/2302.07636)) (Igamberdiev & Habernal 2023) —
  latent-space noise in a seq2seq rewriter; diagnoses the LDP adjacency-constraint noise blow-up. *Fit:*
  shows both the promise (contextual whole-sequence rewrite) and the formal cost of learned DP rewriting
  — frames RD4's guarantee problem.
- **[DP-MLM](../../research-wiki/papers/meisenbacher2024_dp_mlm.md)** ([arXiv 2407.00637](https://arxiv.org/abs/2407.00637)) (Meisenbacher et al. 2024) —
  encoder-only MLM per-token DP rewrite; better utility at low ε than decoder paraphrasing. *Fit:* the
  most on-point RD4 result — an MLM substitutor beats decoder rewriting, supporting "finetune an MLM."
- **Paraphrase-anonymization** (Mattern et al. 2022, the fix half of
  [Limits of Word-Level DP](../../research-wiki/papers/mattern2022_limits_word_level_dp.md), [arXiv 2205.02130](https://arxiv.org/abs/2205.02130)) — a finetuned
  paraphraser with a formal guarantee. *Fit:* the original argument that a learned rewriter beats
  word-level DP on privacy *and* utility *and* fluency simultaneously.
- **MI on MLMs** (Mireshghallah et al. 2022, [2203.03929](https://arxiv.org/abs/2203.03929)) — MLMs leak
  training data via membership inference. *Fit:* the cut-both-ways warning — the substitutor's own model
  is an attack surface, so RD4 must be evaluated against a learned attacker (ties to RD2/F4).

**Cross-cutting connections.** RD4 can absorb RD1 (a model that scores salience while substituting) and
RD2 (a model that reasons about what's sensitive) — the same LLM capability keeps recurring, which is
exactly why it is also the RD5 reconstructor and the RD2 attacker. RD4 handles what RD3 can't: context-
and sense-dependent spans.

**Conclusions.** RD4 is the right tool for premise- and value-bearing spans where meaning depends on
context; the machinery is a finetune away (MLM/infilling), and DP-MLM shows encoder-only MLMs are the
sweet spot at low ε. The unsolved part is a *formal* guarantee for a contextual swap that doesn't
collapse back to per-token metric-DP.

**Open problems.** A privacy guarantee for a context-conditioned substitution; keeping the model small
and on-device; preventing the substitutor from re-leaking via its own memorization (Mireshghallah).

### RD3 vs RD4 — the programmable/learned split

| Axis | RD3 programmable | RD4 learned |
|------|------------------|-------------|
| Context/sense sensitivity | none (type table) | full (attention) |
| Long tail / novel domains | silent rule failure | graceful |
| Formal guarantee | strong (FPE crypto / mDP buckets) | hard (tends back to per-token DP) |
| Reversibility | native (client key) → free de-sanitize | learned guess → needs RD5 |
| Failure mode | brittle but auditable | opaque but robust |
| Best for | identifier & format-bearing spans | premise/value spans in context |

Complementary, routed by RD1: identifiers → RD3 (zero reconstruction cost); premise/value → RD4 (needs
RD5). This split is the concrete payoff of decomposing the scalar ε.

---

## RD5 — Targeted reconstruction architectures

**Framing.** Extraction is **conditional denoising / alignment / infilling** — copy-with-repair over the
perturbed remote generation, conditioned on the local raw document — *not* open-ended decoding. A
**local-only** reconstructor is post-processing, so it consumes **zero** budget and can be arbitrarily
specialized.

**Fixes:** E1, E2. Depends on RD1–RD4 (the substrate defines the corruption to invert).

**Findings**
- **Post-processing immunity** (Dwork & Roth 2014, *Algorithmic Foundations of DP*) — any function of a
  DP output is DP with the same ε. *Fit:* the theorem that makes a local reconstructor *free* — the
  entire economic argument for spending compute on RD5 rather than budget on the channel.
- **BART** (Lewis et al. 2019, [1910.13461](https://arxiv.org/abs/1910.13461)) — denoising autoencoder:
  corrupt, then reconstruct. *Fit:* the extractor's exact objective, and a model pretrained precisely
  for "repair a corrupted sequence conditioned on context."
- **Levenshtein Transformer** (Gu et al. 2019, [1905.11006](https://arxiv.org/abs/1905.11006)) —
  non-autoregressive insert/delete edits with iterative refinement. *Fit:* high copy-rate edit model —
  copy the good tokens, edit only the perturbed ones; the correct inductive bias for copy-with-repair (E2).
- **LaserTagger** (Malmi et al. 2019, [1909.01187](https://arxiv.org/abs/1909.01187)) — cast generation
  as keep/delete/append tags. *Fit:* when output overlaps input heavily (extraction does), tagging is
  faster and higher-fidelity than seq2seq and can't hallucinate the redacted secret as easily.
- **GECToR** (Omelianchuk et al. 2020, [2005.12592](https://arxiv.org/abs/2005.12592)) — sequence-tagging
  error correction, iterative. *Fit:* proves "tag, not rewrite" yields high copy-rate + speed on a
  repair task structurally identical to extraction.
- **Insertion Transformer / EDITOR** ([1902.03249](https://arxiv.org/abs/1902.03249) /
  [2011.06868](https://arxiv.org/abs/2011.06868)) — flexible insertion and repositioning with soft
  lexical constraints. *Fit:* constrained edit generation that can honor typed slots emitted by RD3.
- **Grammar-constrained decoding** ([2502.05111](https://arxiv.org/abs/2502.05111)) — mask decoding to a
  grammar/automaton. *Fit:* enforce typed-slot structure (dates, IDs) during reconstruction without
  retraining — pairs directly with RD3 surrogates.
- **[Split-and-Denoise](../../research-wiki/papers/mai2023_splitanddenoise_protect_large.md)** ([arXiv 2310.09130](https://arxiv.org/abs/2310.09130)) (Mai et al.
  2023) — client noise + client denoise. *Fit:* an existing zero-budget local reconstructor; proof the
  architecture works end to end.
- **[Double-edged Sword](../../research-wiki/papers/meisenbacher2025_double_edged_reconstruction.md)** ([arXiv 2508.18976](https://arxiv.org/abs/2508.18976))
  (Meisenbacher et al. 2025) — reconstruction as adversarial *hardening* post-process. *Fit:* reframes
  RD5 as a privacy tool, not just utility repair — the duality turned productive.
- **[InferDPT extraction](../../research-wiki/papers/tong2023_inferdpt_privacypreserving_inference.md)** ([arXiv 2310.12214](https://arxiv.org/abs/2310.12214))
  (Tong et al. 2023) — the broad-decoder extractor RD5 replaces. *Fit:* the baseline; its E1 flatness
  result is the reason to specialize the architecture.

**Cross-cutting connections.** RD5's job is *defined by* RD1–RD4: it must invert the exact corruption
they produce, deterministically reverse-map the RD3 fraction, and repair only the RD4 residue. It shares
its model class with RD4 (both denoise/infill) and with the RD2 attacker — the recurring LLM capability.
Post-processing immunity is what lets all client-side stages (RD2 detection, RD3 de-sanitize, RD5 repair)
be free.

**Conclusions.** Stop using a general decoder for a denoising task. An edit/tagging model (high copy-rate)
trained on the *exact induced corruption* is the right architecture; it is local, free by immunity, and
harder to make hallucinate the secret than open-ended decoding. Reversible RD3 spans bypass it entirely.

**Open problems.** Training the denoiser on the exact RD1–RD4 corruption so it is a matched inverse, not
a general LM; tuning the copy-vs-repair balance to maximize utility without re-hallucinating redacted
content; evaluating on instruction/QA tasks (E3), which every reconstructor here leaves untested.

---

## Synthesis — one coherent spine

RD0 normalize→resolve spans → RD2 type them (span + QI-combination pass) → RD1 route by role/salience →
RD3 reversible surrogates for identifiers/formats (client-keyed) **+** RD4 contextual swap for
premise/value spans → remote LLM → RD5 matched local denoiser (reverse-map the RD3 fraction, repair the
RD4 residue). **Budget is spent only on the RD4 residue that crosses the wire; RD3 and RD5 are zero-cost
by construction (post-processing immunity).** The single recurring lesson: the hardest sub-problems
(salience, QI reasoning, contextual swap, denoising) all reduce to one LLM capability that is
simultaneously the tool and the adversary — so every component must be built and *evaluated* against
that capability, not against surface metrics.

## Sources

Registered anchors (wiki): [InferDPT](../../research-wiki/papers/tong2023_inferdpt_privacypreserving_inference.md) ([arXiv 2310.12214](https://arxiv.org/abs/2310.12214)),
[Limits of Word-Level DP](../../research-wiki/papers/mattern2022_limits_word_level_dp.md) ([arXiv 2205.02130](https://arxiv.org/abs/2205.02130)),
[Vulnerability of Text Sanitization](../../research-wiki/papers/tong2025_vulnerability_text_sanitization.md) ([arXiv 2410.17052](https://arxiv.org/abs/2410.17052)),
[Reconstruction via LLMs](../../research-wiki/papers/pang2024_reconstruction_dp_text_llm.md) ([arXiv 2410.12443](https://arxiv.org/abs/2410.12443)),
[Double-edged Sword](../../research-wiki/papers/meisenbacher2025_double_edged_reconstruction.md) ([arXiv 2508.18976](https://arxiv.org/abs/2508.18976)),
[CusText](../../research-wiki/papers/chen2022_customized_text_sanitization.md) ([arXiv 2207.01193](https://arxiv.org/abs/2207.01193)),
[SanText](../../research-wiki/papers/yue2021_differential_privacy_text.md) ([arXiv 2106.01221](https://arxiv.org/abs/2106.01221)),
[HaS](../../research-wiki/papers/chen2023_hide_seek_has.md) ([arXiv 2309.03057](https://arxiv.org/abs/2309.03057)),
[Split-and-Denoise](../../research-wiki/papers/mai2023_splitanddenoise_protect_large.md) ([arXiv 2310.09130](https://arxiv.org/abs/2310.09130)),
[DP-Prompt](../../research-wiki/papers/utpala2023_locally_differentially_private.md) ([arXiv 2310.16111](https://arxiv.org/abs/2310.16111)),
[DP-Fusion](../../research-wiki/papers/thareja2025_dpfusion_tokenlevel_differentially.md) ([arXiv 2507.04531](https://arxiv.org/abs/2507.04531)),
[Mahalanobis mDP](../../research-wiki/papers/xu2020_differentially_private_text.md) ([arXiv 2010.11947](https://arxiv.org/abs/2010.11947)),
[Feyisetan calibrated mDP](../../research-wiki/papers/feyisetan2019_privacy_utilitypreserving_textual.md) ([arXiv 1910.08902](https://arxiv.org/abs/1910.08902)),
[d_X curse of dimensionality](../../research-wiki/papers/asghar2024_dxprivacy_text_curse.md) ([arXiv 2411.13784](https://arxiv.org/abs/2411.13784)),
[DP-BART](../../research-wiki/papers/igamberdiev2023_dp_bart.md) ([arXiv 2302.07636](https://arxiv.org/abs/2302.07636)),
[DP-MLM](../../research-wiki/papers/meisenbacher2024_dp_mlm.md) ([arXiv 2407.00637](https://arxiv.org/abs/2407.00637)),
[Lison anonymisation SoK](../../research-wiki/papers/lison2021_anonymisation_models_text.md) ([ACL 2021](https://aclanthology.org/2021.acl-long.323/)).
Other references cited inline with arXiv/DOI/standard IDs.
