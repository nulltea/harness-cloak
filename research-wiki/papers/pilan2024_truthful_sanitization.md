---
type: paper
node_id: paper:pilan2024_truthful_sanitization
title: "Truthful Text Sanitization Guided by Inference Attacks"
authors: ["Ildikó Pilán", "Benet Manzanares-Salor", "David Sánchez", "Pierre Lison"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2412.12928"
  doi: null
  s2: null
tags: ["text-anonymization", "attack-guided", "generalization", "llm", "rd2", "rd4"]
added: 2026-07-02T00:00:00Z
---

# Truthful Text Sanitization Guided by Inference Attacks

## One-line thesis
Replace PII with *truthful generalizations* (broader terms that subsume the original), choosing among
LLM-generated candidates by how well they resist an inference-attack evaluator. Method name: **INTACT**
(INference-guided Truthful sAnitization for Clear Text).

## Method
Two stages, both run on **Mistral 7B Instruct v0.2** (4-bit quantized, multinomial sampling, temperature
0.3, ≤512 new tokens); Llama 3.1 8B Instruct tested as a swap-in for generalizability (§5.8).

1. **Candidate generation.** For each annotated PII span, a one-shot prompt asks the LLM for a list of
   *m* replacement candidates "sorted by order of abstraction" from most to least specific (m=5 in
   experiments). The one-shot example shows a sentence, a target span of the same entity label, and
   possible generalizations. One-shot prompting was chosen because it "yield[ed] the most consistent
   output formatting" vs zero-shot or JSON variants.
2. **Inference-attack selection.** Candidates are examined in generated order (most→least specific). For
   candidate c, the same LLM is prompted to "guess back the original text span based on the rest of the
   text and the replacement", producing *p* guesses (p=5). A guess *matches* if it overlaps the original
   span's lemma set (or n-grams, for named entities; dates require exact lemma overlap). The **first
   candidate none of whose 5 guesses match** is selected. If all m candidates are guessed, fall back to
   the entity-type label (e.g. "organization").
   - **Direct identifiers** (person names, codes) skip generation entirely and get typed placeholders
     (`PERSON_1`) via heuristic rules.

**"Truthful generalization" is operational, not formal.** The paper never defines a subsumption /
entailment relation; it describes replacements as "less specific, but truth-preserving generalizations"
(informally hypernyms) and verifies truthfulness only *ex post* by human annotation and embedding
similarity — no NLI or ontology check inside the pipeline. The LLM is trusted to emit genuine
generalizations, and admittedly sometimes emits definitions/paraphrases instead.

Selection statistics: the *most specific* candidate is chosen for 53% of spans; for ~22% of spans all
five candidates were guessed and the entity-label fallback fired (nearly half of spans for ORG and MISC).

## Evaluation setup
- **Dataset:** Text Anonymization Benchmark (TAB) — 1,268 English ECtHR court-case documents, PII
  annotated by 12 law students (~11% of spans are PII). Test set: 127 documents, >7,300 entity mentions.
  Entity types: PERSON, CODE, ORG, DATETIME, LOC, QUANTITY, DEM, MISC.
- **Privacy metric:** TRIR (Text Re-identification Risk) — a distilbert-base-uncased classifier
  fine-tuned on background-knowledge documents to predict *which individual* an anonymized document
  belongs to; TRIR = re-identification accuracy. This is an adversarial (attack-based) privacy metric,
  though a small fine-tuned one, not a frontier-LLM attacker.
- **Utility metrics:** TPS (Text Preserved Similarity; embedding similarity of replaced spans, weighted
  by BERT information content) and downstream document clustering (NMI). TPS↔NMI Pearson r=0.864
  (p=0.012).

## Key Results
Numbers marked ~ are read off figures (the paper reports TPS/TRIR graphically, not in tables).

| Strategy | TPS | TRIR | NMI |
|---|---|---|---|
| Suppression (remove span) | ~0.78 | ~10.0% | ~0.72 |
| Entity-label replacement | 0.85 | ~9.9% | ~0.80 |
| Presidio (synthetic replace) | ~0.87 | ~9.8% | ~0.82 |
| **INTACT** | **~0.90** | **~10.4%** | **~0.83** |
| Most-specific candidate (no attack filter) | 0.92 | ~12.2% | ~0.82 |

- Headline: INTACT recovers most of the utility lost to suppression (suppression = 22% TPS drop;
  protected spans are "twice as informative" as unprotected ones) at essentially the same
  re-identification risk (all strategies cluster near ≈10% TRIR; most-specific vs suppression differ by
  only 2.2 pp).
- **Manual truthfulness** (600 instances, 2 raters, κ=0.84 truthfulness / 0.76 specificity): INTACT
  93.2% truthful / 81.3% more abstract, vs Presidio 19.7% / 36.2% and Dou et al. 95.2% / 78.3%.
- **Cost:** on one RTX 4000 Ada (20GB), candidate generation for the 7,300-mention test set took ~3 h;
  the inference-attack stage took "approximately 15–20 times more". The attack filter is by far the
  dominant cost.

## Limitations / Failure Modes
Paper's own admissions:
- Generalization quality degrades on **DEM and MISC** (specialized medical/legal terms, foreign words);
  the LLM then produces definitions/explanations rather than generalizations.
- **Entity-label fallback** fires for ~22% of spans overall and ~half of ORG/MISC spans — for those the
  method degenerates to typed-placeholder anonymization.
- Inference-attack selection is **15–20× the generation cost** — expensive per document.
- TPS measures span similarity in isolation, ignoring context, readability, style.

Our analysis (not in paper):
- The attacker used for *selection* is the same 7B LLM as the rewriter, and the attacker used for
  *evaluation* (TRIR) is a fine-tuned DistilBERT. Neither approximates a frontier-LLM
  re-identification adversary; "no guess in 5 matches by lemma overlap" is a weak criterion (a stronger
  attacker with more samples or semantic matching may re-identify what lemma overlap misses).
- TRIR barely separates the strategies (all ≈10%), so the benchmark provides little resolution on the
  privacy axis; the claimed win is almost entirely on utility/truthfulness.
- Direct identifiers get placeholders, so the "truthful generalization" story applies only to
  quasi-identifiers.

## Co-design fitness (doc_orig→doc_p ↔ out_p→out_final)
- **(a) Conditions on / emits:** conditions on the full document context plus the annotated PII span
  (gold TAB annotations — detection is out of scope) and the span's entity label; emits a natural-language
  hypernym/generalization in place of the span (or a typed placeholder for direct identifiers). Output is
  fluent, truthful text — no noise tokens.
- **(b) Client-side record:** the pipeline inherently produces a **span → chosen-generalization map**
  (plus the rejected more-specific candidates, i.e. a partial abstraction lattice per span), though the
  paper never stores or uses it after sanitization. This map is exactly an extractor conditioning signal:
  a truthful generalization is *invertible by narrowing* — the client, holding the original values, can
  map any remote-output mention of "a Scandinavian country" back to "Norway" deterministically. Because
  generalizations subsume the originals, the reverse map is well-typed in a way random-swap (RANTEXT-style)
  outputs are not.
- **(c) Reverse/reconstruction step:** none. Text-release setting; the sanitized document is the product.
- **(d) Round-trip stress:** truthfulness is the key asset here — the remote LLM computes on *true but
  coarser* premises, so task answers remain valid at a coarser granularity rather than being poisoned by
  false facts (Presidio-style synthetic replacement, 19.7% truthful, actively feeds the remote model
  falsehoods). What breaks: (i) tasks whose answer depends on the *suppressed specificity* (exact dates,
  amounts, jurisdictions) lose precision that no local extractor can restore from `out_p` alone — though
  the client-side map can re-specialize mentions it can align; (ii) the ~22% entity-label-fallback spans
  behave like redaction and carry near-zero task signal; (iii) typed placeholders for names survive a
  round trip well (easy reverse map) but leak structure.
- **(e) Privacy vs adversary:** yes, twice — selection-time inference attack (Mistral 7B guess-back) and
  evaluation-time TRIR (fine-tuned DistilBERT re-identification over background knowledge). Both weaker
  than our frontier-LLM re-identification attacker standard.
- **(f) Verdict:** INTACT is the strongest template so far for the *substitutor* half: attack-guided
  candidate selection is exactly "learn/choose the substitution the attacker can't invert", and truthful
  generalization is the property that makes a *local extractor tractable* — narrowing a hypernym back to
  the client's known original is far better-posed than undoing a random or synthetic swap. It contributes
  the selection loop and the invertible-lattice insight; it cannot provide the extractor itself (no
  reverse step exists), is 15–20× too slow in its LLM-guess-back form, assumes gold PII spans, and its
  privacy evidence is against sub-frontier attackers on a benchmark where TRIR barely discriminates.

## Relevance to This Project
**Why surfaced:** round-2 (arXiv) discovery; a strong **RD2∩RD4** node in
[`docs/research/learned-substitution.md`](../../docs/research/learned-substitution.md). **Fit:** realizes
[Lison 2021](lison2021_anonymisation_models_text.md)'s "model disclosure risk, not spans" by *guiding
substitution with an inference attacker* — the attack-guided branch of learned substitution; the
"generalization" surrogate is a middle ground between RD3 typed placeholders and RD4 free rewriting.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Abstract (original summary)

> Text sanitization via *generalizations* — broader but still informative terms that subsume the semantic
> content of the original spans. Instruction-tuned LLMs are used in two stages: generate and rank
> replacement candidates for PII, then evaluate their privacy protection via inference attacks, selecting
> replacements that balance privacy and utility. With Mistral 7B Instruct on the Text Anonymization
> Benchmark, the approach achieves enhanced utility with minimal increase in re-identification risk vs.
> full suppression, and superior truthfulness preservation relative to tools like Microsoft Presidio.
