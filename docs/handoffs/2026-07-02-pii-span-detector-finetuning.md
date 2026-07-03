---
type: handoff
status: current
created: 2026-07-02
updated: 2026-07-02
tags: [detector, gliner, tab, supervised-finetuning, quasi-gap, rd2, handoff]
companion: [../plans/2026-07-02-surrogate-grpo-training.md, ../plans/2026-07-02-d1-prototype-implementation.md]
---

# Handoff: PII span detector finetuning (close the QUASI gap, supervised)

## Task for the next session

Close the detector's measured quasi-identifier recall gap by **supervised finetuning of GLiNER on
TAB gold spans** (or weak labels) — *not* by RL. Detection recall is the pipeline's privacy
ceiling (RD2), and the weak categories are known and measured.

## What the detector is right now

A model + rules union (`src/cloak/detect.py`): **GLiNER small-v2.1** — a 166M zero-shot
span-extraction model taking TAB's 8 categories as free-text labels — union **Presidio**
pattern/checksum recognizers (pure rules, plus added numeric-reference/money recognizers), noise
filters (emoji/pronouns/URL-ellipses/generic temporals), and alias-chain coref (fastcoref is
incompatible with transformers 5.12 — alias chains replaced it). Measured at the P0 gate
(`results/latticecloak_detection_gate.json`, 127 TAB docs): **DIRECT recall 0.998** (excellent,
entity-level 0.997), **QUASI 0.857** with weak spots **DEM 0.56, MISC 0.21, QUANTITY 0.25**. So
there is a trainable model in there, and it has a real measured gap on quasi-identifiers.

## Why the detector stays OUT of the RL policy (decision 2026-07-02, rationale)

The GRPO plans freeze the detector; the rationale is sharper than cost and is a genuine trap:

1. **The RL reward is blind to detection misses — RL would train the detector to under-detect.**
   The utility term rises when spans stay specific; the privacy term (MTI probe) only scores spans
   that *were substituted* — a missed span sits in plain text and costs the reward nothing.
   Gradient direction for a detector inside this loop: detect less, score more. That's not a
   variance problem, it's a reward-hacking channel pointed at the exact thing detection exists to
   prevent. Including it safely would require a document-level privacy head that prices unmasked
   PII (the attack-head escalation in the surrogate plan), which is deliberately deferred.
2. **Detection has something strictly better than RL: labels.** TAB gold spans exist; supervised
   finetuning of GLiNER on TAB (or weak labels) is cheaper, higher-signal, and safe. RL earns its
   complexity only where no supervision exists — the ranker/infiller, not the detector.
3. Project philosophy treats detection recall as the **measured privacy ceiling** (RD2 evidence),
   reported, not patched from inside the method.

**Revisit trigger:** if the attack head escalates to the document-level encoder
(fork 1 escalation in
[`2026-07-02-surrogate-grpo-training.md`](../plans/2026-07-02-surrogate-grpo-training.md)), misses
become priced by the reward and detector-in-the-loop stops being a hacking channel — at that point
it's a legitimate future fork.

## Scope and evaluation for the finetuning itself

- **Target:** QUASI recall on the weak types (DEM 0.56, MISC 0.21, QUANTITY 0.25) without
  regressing DIRECT recall 0.998 or precision (spurious spans coarsen premise text → utility cost;
  the P0 gate script `scripts/latticecloak_detection_gate.py` is the harness — rerun it as the
  before/after measure, same 127-doc TAB test split).
- **Data:** TAB gold spans (`corpora/tab/`, standoff JSON; 1,268 ECtHR docs, 8 types ×
  direct/quasi × `entity_id` chains). Mind the P0 caveat: one regex was tuned on 5 test docs —
  keep train/dev/test splits clean for the finetune.
- **GLiNER finetuning:** upstream repo has training scripts (span-level BIO-free matching;
  gliner pins transformers<5.7 but runs fine on 5.12 — see primer gotcha). LoRA or full finetune
  of the 166M model fits the iGPU easily. Keep the zero-shot label-text interface so
  `detect.py` needs no API change; the finetuned checkpoint is a drop-in `model=` swap.
- **Honesty rules:** report per-type recall/precision before/after; the detector gap is a
  *finding* (RD2) as well as a fix — don't silently absorb it. Domain shift caveat: TAB is ECtHR
  legal text; the task corpora are clinical/email — check transfer on a small hand-labeled sample
  of those before trusting the gain off-TAB.

## Repo state at handoff (context)

Active work: the surrogate-reward GRPO plan is finalized
([`2026-07-02-surrogate-grpo-training.md`](../plans/2026-07-02-surrogate-grpo-training.md) — read
it first: policy = ranker (frozen-roberta features + MLP, option 1) then + infiller; detector
frozen; task corpora = clinical + email). Implementation of the surrogate reward module +
validation gate was just starting (nothing committed for it yet). This detector finetune is
**independent, parallel work** — it touches `detect.py`'s model checkpoint only; coordinate on the
one-GPU rule (`CLAUDE.md`) with any training runs, and re-run the P0 gate + a τ-sweep smoke after
swapping the checkpoint so downstream numbers stay comparable.

## Suggested skills for the next session

`/grill-me` (scope the finetune: LoRA vs full, label mapping, split policy), perf gate via
`/auto-review-loop` with `scripts/harness/perf_gate.md` before the training run,
`/result-to-claim` for the before/after gate numbers.
