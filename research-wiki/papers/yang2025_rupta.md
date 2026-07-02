---
type: paper
node_id: paper:yang2025_rupta
title: "Robust Utility-Preserving Text Anonymization Based on Large Language Models"
authors: ["Tianyu Yang", "Xiaodan Zhu", "Iryna Gurevych"]
year: 2025
venue: "ACL 2025"
external_ids:
  arxiv: "2407.11770"
  doi: null
  s2: null
tags: ["llm-anonymization", "adversarial", "distillation", "re-identification", "rd4", "rd2"]
added: 2026-07-02T00:00:00Z
---

# RUPTA: robust utility-preserving text anonymization with LLMs

## One-line thesis
Anonymize against LLM re-identification with three collaborating LLM components — a privacy
evaluator, a utility evaluator, and a lexicographic optimizer — then distill the pipeline into a
lightweight local model (SFT + DPO on the optimization traces).

## Method

Iterative black-box optimization loop, up to **T = 5** iterations, all components LLM-prompted:

- **P-Evaluator (privacy):** prompted to infer the person from the anonymized text, returning
  **top-K re-identification candidates** (default **K = 10**). If the ground-truth identity
  appears, the privacy score is its **rank r** among the K candidates, plus *textual feedback*
  naming the clues that enabled the inference; if absent, score = K+1 (max protection).
- **U-Evaluator (utility):** outputs a confidence score that the anonymized text still links to
  its ground-truth downstream label (occupation), i.e. how much task-relevant signal survives.
- **Optimizer (lexicographic, privacy first):** while privacy score < K+1, follows an
  "anonymize further" instruction `I_pr` conditioned on the P-Evaluator's textual feedback;
  once privacy ≥ K+1, switches to a utility-repair instruction `I_ur` ("optimize utility without
  compromising achieved privacy"). Terminates when both objectives hit preset maxima or T rounds.

**Component models:** GPT-4 as P-Evaluator, U-Evaluator, and Optimizer in the main setup;
ablations swap in GPT-3.5, Llama-3-70b, Mixtral 8×22b for the optimizer / P-Evaluator.

**Distillation to local models:** students **Llama-3-8b** and **Phi-3 Mini**. Recipe: SFT with the
teacher's *final* anonymization result as the label, then **DPO using intermediate vs final
optimization results as the preference pairs** (later iterations preferred). Motivated by cost:
RUPTA with GPT-4 averages 76.38 s per paragraph, ~3,846 prompt + ~697 completion tokens per sample.

**Cost of the loop:** every document costs up to T rounds × 3 GPT-4 calls; distillation is the
paper's answer, not a cheaper loop.

Code/data: <https://github.com/UKPLab/acl2025-rupta>.

## Key Results

**Datasets:**
- **DB-bio** (new, released): celebrity biographies from the DBpedia Classes dataset with
  occupation labels; 1,938 train / 243 val / 239 test; biography length filtered to 200–700 words.
- **PersonalReddit (PR):** 525 human-verified synthetic Reddit comments (207 test) with
  occupation labels and implicit attributes (gender, location).

**Metrics:** privacy = **SR** (success rate: % of cases where ground truth appears in the LLM's
top-K inferences, lower better) and **CS** (LLM-stated confidence linking text to the person,
lower better); utility = Accuracy/F1 of a **BERT-based occupation classifier fine-tuned on the
original training data**, applied to anonymized test text.

DB-bio test (GPT-4 components):

| Method | SR↓ | CS↓ | Acc↑ | F1↑ |
|---|---|---|---|---|
| Original text | 100.0 | 98.45 | 99.58 | 99.61 |
| AF (adversarial-feedback baseline, Staab-style) | 52.91 | 50.84 | 92.02 | 91.75 |
| **RUPTA (GPT-4)** | 52.67 | 53.11 | **96.02** | **95.91** |

- Privacy is ~tied with AF (52.67 vs 52.91 SR); the win is utility: relative utility loss 0.1618
  vs AF's 0.4048.
- PR test: RUPTA SR 35.75 / Acc 35.75 vs AF SR 35.40 / Acc 21.26; note PR original text has
  SR only 49.76 and Acc 58.45 — small headroom, task is hard even unanonymized.
- Human eval (100 samples): 3.96/5 semantic preservation, 3.68/5 fluency — best among baselines
  incl. Azure and AF.

Distillation (DB-bio):

| Model | SR↓ | CS↓ | Acc↑ |
|---|---|---|---|
| Llama-3-8b (raw) | 69.38 | 71.56 | 95.40 |
| Llama-3-SFT | 56.90 | 59.97 | 96.33 |
| **Llama-3-DPO** | **51.03** | 53.78 | 93.28 |
| Phi-3 (raw) | 71.42 | 74.49 | 93.22 |
| Phi-3-SFT | 60.25 | 65.73 | 97.48 |
| Phi-3-DPO | 54.15 | 55.90 | 92.15 |

- Llama-3-8b+DPO essentially matches the GPT-4 teacher on privacy (SR 51.03 vs 52.67) with a
  ~2.7 pt utility drop (Acc 93.28 vs 96.02). DPO on intermediate-vs-final traces is what closes
  the privacy gap that SFT alone leaves (56.90 → 51.03).

## Limitations / Failure Modes

Paper's own admissions (Limitations section): computationally intensive (GPT-4 loop); DB-bio
celebrity biographies "may not fully represent the variety of scenarios"; **static adversary
assumption** (real attackers evolve); **no formal privacy guarantee** ("offering a formal
guarantee for NLP-based anonymization methods remains challenging").

Additional analysis (ours):
- **Evaluator = attacker circularity:** the same GPT-4 that scores privacy during optimization
  defines the SR/CS evaluation; the loop optimizes each document *directly against the
  evaluation attacker* until it exits its top-10. The paper does not clearly report SR under an
  attacker independent of the optimization loop — realized privacy against a different or
  stronger re-identifier is unknown.
- **Celebrity re-identification ≠ private-individual linkage:** SR measures whether GPT-4's world
  knowledge recalls a *famous* person; protection of a private individual (linkage across their
  own documents/attributes, the Staab RD2 threat) is only partially exercised by PR.
- **Rank-in-top-K is a coarse privacy signal:** score K+1 means "fell out of this attacker's
  top-10", not "unidentifiable"; K is a knob of the evaluator, not of realized privacy.
- **Per-document stopping rule** yields uneven realized privacy across documents (each stops as
  soon as it beats *this* attacker), so corpus-level SR mixes operating points.
- Utility is a single label-level task (occupation classification); semantic content beyond the
  label is only covered by the 100-sample human eval.

## Co-design fitness (doc_orig→doc_p ↔ out_p→out_final)

- **(a) Conditions on / emits:** optimizer conditions on the current full text plus the
  P-Evaluator's *natural-language leak diagnosis* (which clues identified the person) and,
  in phase 2, a utility-repair instruction; emits a full free-form rewrite each round. No
  span-level substitution structure.
- **(b) Client-side record:** no reverse map. Intermediate iterations are retained — but only as
  DPO preference data for distillation, not as an edit trace; nothing links doc_p spans back to
  doc_orig spans.
- **(c) Reverse/reconstruction step:** none, and nothing is local in the main pipeline — all
  three components are GPT-4 API calls, so doc_orig itself is sent to a remote LLM during
  anonymization. Only the *distilled student* is local. There is no out_p→out_final concept.
- **(d) Round-trip behavior:** partially better positioned than pure text-release work — utility
  is already a *task computed on doc_p* (occupation classification), so the framework's
  U-Evaluator slot could hold a round-trip utility instead. But the lexicographic scheme repairs
  utility only for one fixed label; a remote task needing entities the optimizer generalized away
  gets no repair signal, and with no reconstruction stage the degradation is unrecoverable.
- **(e) Adversary:** yes — an LLM re-identification attacker (top-K person inference, GPT-4),
  which is exactly this project's privacy-measurement framing; but see the circularity caveat —
  attacker and in-loop evaluator are not independent.
- **(f) Verdict:** contributes two directly reusable pieces: (1) the *evaluator-feedback loop* —
  a privacy evaluator that returns a rank **plus a textual leak diagnosis** the rewriter
  conditions on is a strong iterative substitutor pattern, and its lexicographic
  "privacy-first, then repair utility" schedule maps cleanly onto "reach target realized privacy,
  then maximize round-trip utility"; (2) the *distillation recipe* (SFT on final outputs + DPO on
  intermediate-vs-final traces) as the concrete route from an expensive teacher loop to our local
  substitutor, with numbers showing DPO is the step that preserves teacher privacy. It cannot
  provide: any extractor/reconstruction design, a leakage-free anonymization stage (teacher sends
  doc_orig to GPT-4 — unusable as-is for our threat model), or privacy numbers valid against an
  attacker outside its own loop.

## Relevance to This Project
**Why surfaced:** a **2025 SOTA** on the *non-DP LLM-adversarial* branch of RD4 in
[`docs/research/learned-substitution.md`](../../docs/research/learned-substitution.md). **Fit:** the
strongest empirical privacy-utility trade-off (no formal DP), directly confronting the LLM QI-reasoning
re-identifier (RD2/[Staab](staab2024_llm_anonymizers.md)); distillation-to-local is the practical route
to an on-device RD4 substitutor.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Abstract (original)

> Anonymizing text that contains sensitive information is crucial for a wide range of applications.
> Existing techniques face the emerging challenges of the re-identification ability of large language
> models (LLMs), which have shown advanced capability in memorizing detailed information and reasoning
> over dispersed pieces of patterns to draw conclusions. When defending against LLM-based
> re-identification, anonymization could jeopardize the utility of the resulting anonymized data in
> downstream tasks. In general, the interaction between anonymization and data utility requires a deeper
> understanding within the context of LLMs. In this paper, we propose a framework composed of three key
> LLM-based components: a privacy evaluator, a utility evaluator, and an optimization component, which
> work collaboratively to perform anonymization. Extensive experiments demonstrate that the proposed
> model outperforms existing baselines, showing robustness in reducing the risk of re-identification
> while preserving greater data utility in downstream tasks.
