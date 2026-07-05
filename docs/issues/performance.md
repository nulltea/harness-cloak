---
type: dev-log
status: current
created: 2026-07-05
updated: 2026-07-05
tags: [performance, detection-gate, gliner, presidio, checkpoint-selection]
---

# Detection-gate checkpoint-selection sweep is ~5–7× slower than necessary

## Symptom
Selecting a fine-tuned detector checkpoint runs `scripts/latticecloak_detection_gate.py` across a grid
(3 epoch checkpoints × 5 thresholds = 15 cells) on TAB dev (127 docs). Each cell's detection loop is
**46–90 s** (`wall_s`, excludes model/Presidio load); the full 15-cell sweep is **~17 min** on the base
model (worse on large). Observed **~100 % on a single CPU thread** during runs.

## Measured breakdown (base model, 127 dev docs, gfx1151, GPU idle otherwise)
Profiled one cell by timing each stage of `Detector.detect` over the corpus:

| stage | time | share |
|---|---|---|
| **GLiNER** (`batch_predict_entities`, per-doc) | 49.2 s | **73 %** |
| **Presidio** (`analyze`, spaCy `core_web_lg` full pipeline) | 18.4 s | 27 % |
| chunking + overlap-matching loops | ~0 s | ~0 % |

Sub-measurements:
- **Presidio is NOT the dominant cost** (initial guess was wrong). Presidio over all 127 docs = **13.4 s**
  standalone (106 ms/doc), of which **spaCy is 11.6 s (86 %)**. spaCy loads the full pipeline
  (`tok2vec, tagger, parser, attribute_ruler, lemmatizer, ner`) though Presidio only needs `ner` (+tokenizer).
- **GLiNER cost is per-chunk, not per-call overhead.** 127 docs → **655 chunks** (~5.2/doc), currently issued
  as **127 tiny per-doc `batch_predict_entities` calls**. Flat-batching all 655 chunks in one call is only
  **~17 % faster (49.2 s → 40.9 s)** — so the cost is intrinsic per-chunk work: span enumeration
  (seq ~250 tok × `max_width` × 8 labels) + **single-threaded CPU tokenization** (`WordsSplitter` /
  `prepare_batch`) on a modest iGPU whose forward pass is fast.
- **The single-thread-100 % is the CPU-side serial work** — GLiNER's tokenization/prepare + spaCy — not the
  GPU forward.

## Root cause of the *sweep* slowness
The threshold only **post-filters GLiNER span scores**; it does not change Presidio at all, and does not
change the GLiNER forward pass (only the decode cutoff). Yet each of the 15 cells re-runs the **entire**
pipeline:
- **Presidio** is **checkpoint- AND threshold-invariant** — it depends only on the corpus — but runs **15×**.
- **GLiNER** is **threshold-invariant at the score level** — it should run **once per checkpoint (3×)** — but
  runs **15×**.

So ~15× full-pipeline work where **1× Presidio + 3× GLiNER** suffices.

## Proposed optimizations (ranked by measured payoff)

1. **Sweep mode in the gate — the real win (~5–7×).** Add a mode that, per checkpoint, runs GLiNER **once**
   at threshold 0 (or the lowest grid point) and keeps the raw per-span scores; runs Presidio **once per
   corpus** and caches its spans; then computes recall/precision at **every threshold from the cached
   scores in memory**. Turns 15 × ~67 s ≈ **17 min → ~3 min** (3 GLiNER passes + 1 Presidio pass + cheap
   in-memory thresholding). Sketch:
   ```
   presidio_spans = analyze_corpus(docs)            # once per corpus
   for ckpt in checkpoints:
       raw = gliner_scores(docs, ckpt)              # once per checkpoint (keep score per span)
       for thr in thresholds:
           preds = [s for s in raw if s.score >= thr] + presidio_spans
           metrics[ckpt, thr] = score_against_gold(preds, gold)   # in-memory, ~0s
   ```
   This is the fix worth building — we have now swept 3 times (v3-large, base+mix), which is past the point
   where it pays off.

2. **Trim the spaCy pipeline (~2× on Presidio's 27 %).** Load spaCy with
   `exclude=["parser","tagger","attribute_ruler","lemmatizer"]` (Presidio's default recognizers need only
   `ner` + tokenizer). Roughly halves the 18 s Presidio stage. Verify no recognizer depends on POS/lemma
   first (the default `SpacyRecognizer` uses only NER entities).

3. **Cross-document GLiNER batching (~17 %, minor).** Flatten all corpus chunks and batch across docs
   instead of per-doc calls. Small win; only worth doing inside the sweep-mode refactor, not standalone.

4. **(Optional) smaller spaCy model** (`core_web_sm` vs `lg`) — faster NER, some recall cost on Presidio's
   PERSON/LOC; measure before adopting (Presidio is only a precision-side union member here).

## Notes / scope
- Numbers are base model (deberta-v3-small) on gfx1151. The **large** backbone's GLiNER stage is ~10× slower
  per chunk (24 vs 6 layers, `max_width` 100 vs 12) — sweep mode matters even more there. Large's
  `max_width=100` also inflates the span-enumeration cost for spans the dataset caps at 60 words; capping
  inference `max_width`≈60 would cut it (see the v3 training record).
- The overlap-matching Python loops are negligible (~0 s) — not worth optimizing.
- No correctness impact from any of the above; sweep mode must reproduce the current per-cell numbers exactly
  (regression-check one cell against the looped result).

## Origin
Surfaced while sweeping checkpoints for the v3 (large) and base+generality-first detector runs
(`research-wiki/training/2026-07-04-FT-detector-v3-large-balanced.md`,
`research-wiki/training/2026-07-05-FT-detector-v4-base-genfirst-mix.md`).
