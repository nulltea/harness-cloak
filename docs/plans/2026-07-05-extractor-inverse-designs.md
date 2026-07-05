---
type: plan
status: current
created: 2026-07-05
updated: 2026-07-05
tags: [extractor, invert, reconstructor, semantic-window, edit-tagger, seq2seq, detector-aligned, post-processing]
companion: [../handoffs/2026-07-05-rl-pilot-runbook.md, ../research/learned-substitution.md, ../research/learned-PII-detection.md, 2026-07-05-detector-pointer-extractor.md]
---

# Extraction (`invert()`) designs: out_p → out_final

## Context

The rule extractor (`src/cloak/extract.py`: placeholder swap-back + exact/fuzzy-90
generalization narrowing) inverts **36%** of level fills today (exact 20% + fuzzy90 16%,
`results/extractor_miss_audit.json`, 75 fills). The 60–90 fuzzy band is 44% but mostly
spurious (cos 0.02–0.15 generic fills); the **truly recoverable paraphrase band is ~5–8%**;
20% is absent (absorbed by the remote answer — no extractor can help). Placeholder echo
23.9% vs ~36% level: the reward bias favors specificity, so the rule extractor was
consciously kept for the RL pilot.

All three designs run **locally** on out_p + R (the substitution record), so by
post-processing immunity they carry **zero privacy cost** — privacy is fixed at doc_p.
The comparison metric is **utility of out_final at identical upstream settings**, plus the
extractor-gaming guard below. Designs 1–3 are ordered by cost; each subsumes the previous
as its fallback (a later design only fires where the earlier one reported a miss), so they
compose into one cascade rather than compete. Design 4 (detector-aligned) is a **standalone
evaluation arm** — it inverts the search direction rather than extending the cascade, and is
compared against the cascade at identical settings.

**Shared re-gate rule** (handoff): ANY extractor change invalidates cached anchors,
validated probes, scan verdicts, and trained policies together. Land extractor changes
immediately **before** a probe build; budget one full probe+scan rebuild per landing.

**Shared gaming guard**: a smarter extractor lets the policy learn "vague fills that the
extractor rescues". Monitor the **exact-vs-(fuzzy+semantic) recall gap per checkpoint**
during any RL run; a widening gap = the policy is leaning on the extractor, report it.

## Design 1 — semantic-window matcher (rule extension, zero training)

Pre-registered in `extract.py`'s own docstring. Targets the 5–8% recoverable band only.

**Mechanism.** When exact and fuzzy-90 both miss for a generalization fill:
1. Locate candidate windows by rapidfuzz alignment in the 60–90 band (the audit's
   `band60_90` classifier is exactly this code — lift it from
   `scripts/spikes/extractor_miss_audit.py`, snap to word/sentence boundaries).
2. Score cosine(fill, window) with all-MiniLM-L6-v2 (already used by the audit; local).
3. Accept iff cos ≥ τ **and** a type-sanity check passes: the window is the same
   detector type as the original span (dates align to dates, quantities to quantities —
   cheap regex/type-lexicon check, not a model). Then replace the aligned slice with the
   original surface, as the fuzzy path does.
4. New stat key `gen_semantic`; absent otherwise unchanged.

**τ selection.** The audit shows spurious matches at cos 0.02–0.15; expect a wide margin
to true paraphrases. Pick τ on the 75 audited fills (they are labeled by the audit run):
choose the highest τ with zero false accepts, report the recall it buys. If the margin is
narrow (true paraphrases below ~0.5), that is a finding — the band is not separable and
Design 1 caps out; report, don't tune.

**Eval & success.** Re-run `extractor_miss_audit.py` with the new path: success = it
converts most of the true-recoverable 5–8% to inverted with **zero** new false inversions
on the spurious band. Failure (can't separate) is a legitimate result and directly
motivates Design 2/3.

**Cost.** ~½ day. MiniLM inference per missed fill only (a few per doc). No training run,
no training record needed; one probe+scan rebuild to re-gate.

**Risks.** False accepts silently corrupt out_final with wrong surfaces (worse than a miss
— it asserts wrong facts to the user). Hence zero-false-accept τ selection and the
type-sanity gate. Ambiguous multi-window ties: take best-cos only, require a margin over
the runner-up.

## Design 2 — edit-tagger reconstructor (small supervised model, no free generation)

An **efficient, tailored architecture** per the project positioning: a token-level tagger
over out_p that decides *where* to invert; substitution itself stays a copy from R. The
model never generates free text, so it cannot hallucinate content — it can only point.

**Mechanism.** Encoder — **init pre-registered as the latest FT-detector checkpoint**
(span-pointing transfer; ablation: base-encoder init at equal data, to measure what the
transfer buys) — reads `[out_p ; linearized R]` and tags each out_p token with either
`KEEP` or `INVERT→entry_i` (a pointer into R's entries, implemented as per-entry span
scoring: for each R entry, predict the start/end of its mention in out_p or "absent" —
the same extractive-QA head shape as span detection, one query per R entry). Tagged spans
are replaced by the entry's original surface; everything else is copied verbatim.

**Training data story.** The seed corpus pre-exists: every round trip already run (pilot
env builds, support scans, the audit) yields (out_p, R) pairs. Labels:
- **Silver positives**: exact/fuzzy-90/Design-1 hits — spans the rule extractor already
  aligns (free, abundant).
- **Hard positives**: paraphrase-drift mentions. Generate synthetically: take a doc_orig,
  substitute, and ask the local pin (gemma E4B) to *restate* out_p sentences containing
  fills — the fill's drifted mention is known by construction. No remote calls, no new
  privacy surface.
- **Negatives**: the audit's spurious band60_90 windows (labeled "absent") — exactly the
  confusions Design 1 struggles with.

**Selection & eval.** Held-out docs from a corpus not in training (train clinical → eval
lexsum and vice versa; the open-label generality requirement). Metrics: inversion
recall/precision vs the audit labels; end-to-end utility of out_final vs Design 1 at
identical settings. Precision floor ≥ the Design-1 operating point — a rescued mention is
worthless if bought with wrong-surface corruption.

**Cost.** ~2–3 days incl. data build; one FT run (training record
`FT-extractor v1` per the schema, spec-then-results). Local GPU, small encoder — fits the
one-GPU-process rule easily.

**Risks.** Silver-label noise teaches it the rule extractor's blind spots; mitigate with
the synthetic hard positives. Pointer head can't handle a mention that *fuses two fills*
into one span (rare; count it in the audit before caring). Absent detection must be
reliable or it force-points at noise — keep an explicit "absent" outcome per entry with
its own threshold.

## Design 3 — denoise seq2seq reconstructor (learned rewrite, the project-goal endgame)

The full replacement from the project direction: a local denoise/edit model that rewrites
out_p into out_final conditioned on R, able to fix what pointing cannot — fused mentions,
grammatical fallout of narrowing (agreement, articles), and partially-absorbed content
where the fill's *consequence* appears but its surface does not.

**Mechanism.** flan-t5-base + LoRA (pre-registered in `extract.py`'s docstring; swap in a
newer small seq2seq only with a measured reason). Input: `out_p [SEP] R-as-text`
(entries linearized `replacement → surface`, typed). Output: out_final. Decode with
**copy-biased constrained decoding**: candidate original surfaces from R are the only
novel strings allowed to enter; otherwise decoding is constrained to out_p's vocabulary
window (levenshtein-style edit beam). This keeps the hallucination surface at
Design-2 level while allowing free *edits around* the substitutions.

**Training data story.** Same seed corpus as Design 2 plus a synthetic denoise objective
that needs no labels at all: take any local document y (clinical/lexsum train split),
apply the real substitutor to get y_p and R, optionally paraphrase y_p locally (gemma
pin), train the model to recover y from (y_p, R). This is the classic denoising recipe —
the substitutor itself is the corruption process, so train data is unlimited and exactly
on-distribution for the mechanism (though y_p lacks the remote model's answer-style
drift; the round-trip pairs from the seed corpus cover that gap and get upweighted).

**Selection & eval.** The only comparison that counts: utility of out_final at matched
realized privacy vs the rule cascade, per corpus, never averaged. Plus: (a) faithfulness
probe — NLI or fact-QA between out_p-with-gold-inversion and model output, since a
rewriter can drop or distort content a pointer never touches; (b) ph_residue must stay 0;
(c) the gaming-gap guard, which matters *most* here (the strongest extractor is the
strongest gaming channel). Training record `FT-extractor v2` (or v1 if Design 2 is
skipped), spec before run.

**Cost.** ~1 week incl. data pipeline and eval harness; the data build is the bulk.
LoRA on t5-base is cheap on the local GPU.

**Risks.** Hallucination/omission despite constrained decoding — the faithfulness probe
is a hard gate, not a diagnostic. Memorization of training docs leaking into outputs for
*other* docs (the substitutor-memorization leak channel from the honesty rules): eval
docs must be disjoint from train and checked for train-doc n-gram intrusion. Latency: a
seq2seq pass per document vs Design 1/2's per-miss scoring; if latency matters, run it
only on docs where the cascade reported misses.

## Design 4 — detector-aligned extractor (standalone arm, typed assignment)

Uses the pipeline's own span detector (`src/cloak/detect.py`, 8-type TAB schema — the same
types R entries carry from substitution time) to aid extraction. Zero new training for the
core mechanism.

**Mechanism.** One detector pass over out_p produces typed spans. Inversion becomes a
**typed one-to-one assignment problem** instead of text search:
1. For each generalization entry in R, the candidate set is the detected spans of the
   **same type** (placeholder swap-back unchanged, runs first; detected spans consumed by
   it are excluded).
2. Score each (entry, span) pair by MiniLM cosine(fill, span text), fuzzy score as
   tiebreak. Greedy one-to-one assignment (R is injective), accept iff score ≥ τ **and**
   the winner clears a margin over the runner-up same-type span; otherwise abstain.
3. Accepted assignments replace the detected span with the original surface; unassigned
   R entries are `absent`. Stat key `gen_detector`.

This inverts the search direction of Designs 1–3: instead of "where does this fill appear
in out_p?", it asks "which sensitive mentions did the remote answer produce, and whose are
they?" A generic window with no detected typed span is never a candidate — which is what
kills the spurious 60–90 band by construction rather than by threshold.

**Checkpoint — deferred decision, rule pre-registered.** Gating spike before any build:
run production `detect()` (GLiNER∪Presidio zero-shot) AND the latest FT-detector
checkpoint over the audit's 75 labeled fills' out_p; select by **recall on drifted
mentions at a fixed precision floor**. The FT track trained on source-document text;
out_p is remote-answer prose — if both checkpoints miss badly on that distribution, the
design caps out. Report the cap as a finding; do not tune around it.

**Learned upgrade — backbone transfer.** The detector connection extends to Design 2:
its encoder init is pre-registered as the FT-detector checkpoint (see Design 2), so
Design 4's gating spike doubles as evidence for whether span-pointing skill transfers to
answer-style prose at all.

**Eval & success.** Own arm on the audit harness: inversion recall/precision vs the rule
extractor and vs Design 1 at identical settings; end-to-end out_final utility; shared
gaming guard and re-gate rule apply. Success = it beats Design 1 on the recoverable band
at equal-or-better precision, or the gating spike shows the detector can't see drifted
mentions (also a result — it prices the FT-detector's distribution generality).

**Cost.** Gating spike ~hours (audit artifacts exist; detector inference is local).
Mechanism ~1 day. One detector pass per doc at extraction time — cheap, local.

**Risks.** Detector distribution shift (the gating spike is the gate). Detector false
positives force wrong assignments — the precision floor and abstain margin are hard
constraints, not diagnostics. Multiple same-type spans with close scores (e.g. two dates)
— abstain rather than guess; count abstentions separately from absents. Type disagreement
(detector types a drifted mention differently than the original span) shows up as a miss;
the gating spike's recall metric must be computed per-type to expose it.

## Decision order

Run Design 1 first — it is nearly free and its measured result (separable band or not)
decides how much of Designs 2/3's motivation survives. Design 4's gating spike runs
alongside it (same audit artifacts, hours of local inference) and additionally decides
Design 2's init question before any FT run is specced. Design 2 vs 3 is not either/or:
Design 2's pointer head is the low-risk step and its eval harness is reusable as
Design 3's baseline; go straight to Design 3 only if the audit shows fused/grammar-drift
misses (which pointing cannot fix) are a material share of the recoverable band.
