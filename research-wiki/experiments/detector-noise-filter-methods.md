---
type: experiment
node_id: exp:detector-noise-filter-methods
title: "Detector-noise post-filters: threshold, kNN-OOD, NLI (assertion/substitution), skweak"
idea_id: "idea:semantic-detector-noise-gate"
verdict: no
confidence: high
date: "2026-07-11"
result: "No cheap post-hoc signal separates real detected entities from noise: detector-confidence AUC 0.63, kNN-OOD (ID=profile) null, NLI-assertion 0.677, NLI-substitution 0.733 — all unusable at privacy-safe thresholds (<=10% real-loss -> <=16% junk). Margin layer measured-unsafe (drops ~30 real). skweak not adopted. Floor = deny-list + exact link; real fix is upstream detector."
hardware: "AMD Strix Halo iGPU (gfx1151), host .venv torch"
duration: "~1 session of spikes"
provenance: "scripts/spikes/{knn_ood_filter,nli_noise_filter,nli_selftype_filter,nli_subst_filter}_spike.py, scripts/spikes/skweak_gate_spike.py; span source results/mined_lattice_profile_spans_large.jsonl (2770 spans, 1287 unique); eval built from profile-linked=real / deny-list=junk"
companion: "docs/specs/detector-noise-semantic-gate.md, docs/issues/2026-07-10-detector-junk-and-noise-gate-limits.md"
added: 2026-07-11T00:00:00Z
tags: ["detector", "noise-filter", "open-set", "nli", "knn-ood", "weak-supervision", "empirical-honesty"]
---

# Detector-noise post-filters: threshold, kNN-OOD, NLI (assertion/substitution), skweak

**verdict:** `no` (no cheap post-hoc signal separates real detected entities from noise — six methods measured, all unusable at privacy-safe thresholds)  ·  **confidence:** `high`  ·  tests `idea:semantic-detector-noise-gate`

## Question

A zero-shot GLiNER miner over-generates: of ~200 surviving `injury` entries, ~half are non-entity noise, out-of-scope entities (anatomy, objects), or wrong-type routings. The rule-based `is_noise_span` deny-list fails open (only removes enumerated families). Can a cheap post-hoc filter — the detector's own confidence, embedding-geometry OOD, or NLI type-verification — separate real entities from noise well enough to ship? Bar: must beat the deny-list without dropping real (privacy-sensitive) entities; a runtime false-drop is a leak.

## Metrics

Common eval (representative, real detected spans w/ context): **eval-KEEP** = span surface links to the profile under its own type (known real, n≈310); **eval-DROP** = surface in a deny-list junk set (known junk, n≈32). AUC = entailment/score separating junk-vs-real; operating point = junk-dropped at bounded real-loss. Detector reference = 324 ms/doc, 24.9 spans/doc (GLiNER-base).

| Method | Separation | Best operating point | Cost | Verdict |
|---|---|---|---|---|
| **Raise detector threshold** | AUC **0.63** | t=0.4: 21% junk / 11% real; t=0.5: 32% / 23% | free | blunt; real entities span 0.32–0.96 (chemo 0.40, radiation 0.96) → no clean cut |
| **Margin layer (shipped Layer 4)** | — | drops 41 "noise", **~30 are real** entities | cheap | **unsafe** (measured false drops); disable |
| **kNN-OOD, ID=profile (Sun 2022)** | — | rejection P/R **0.0 / 0.0** at ≤0.1% fdr; drops 1/1262 | cheap | **null** — profile is not a separable ID cloud for detected spans |
| **NLI v1 (11-hyp argmax, τ=0.5)** | — | drop 607 / retype 257 / keep 383 | 4.8× detector | verbalization-broken: generic target + distractor phrases both over-fire; recovers 16 real |
| **NLI v2 (self-type assertion, calibrated)** | AUC **0.677** | 10% real → 16% junk; Youden τ=0.86 → 94% junk / **59% real** | **0.08–0.12×** detector | ill-posed query (see Reasoning); recovers 37/39 margin-dropped real; no usable τ |
| **NLI v3 (substitution frame, `nli_gate`)** | AUC **0.733** | 10% real → 0% junk; 20% → 44%; Youden τ=0.66 → 78% junk / **39% real** | **0.11–0.15×** detector | well-posed query, best of all (+0.06 over assertion) but still unusable at safe τ; 193/1287 spans emptied by nli_gate prep filters (confound) |
| **skweak NaiveBayes aggregation** | — | fused drop-recall 0.696 vs layered 0.739 @ equal precision | cheap fit | **not adopted** — Naive-Bayes independence violated by correlated LFs |

## Reasoning

**No cheap signal separates, and each fails the same way — the noise and the real entities are genuinely entangled in every cheap feature.** Detector confidence is miscalibrated on broad zero-shot labels (confident junk + unconfident real coexist: `stroke` 0.87 wrong, `chemotherapy` 0.40 right), so AUC 0.63. Embedding-geometry (kNN-OOD) with ID=profile is a **null result**: held-out real profile surfaces sit at the same k-NN distances from the profile as junk, so any threshold sparing reals spares junk — the current margin layer only "worked" by dropping ~30 real entities, and its junk list did the real (bad) work. NLI **v2 is confounded by an ill-posed query**: the hypothesis "X is a &lt;type&gt;" is checked against the span's *usage* sentence, but usage sentences never assert an entity's category — measured directly: `cocaine`→0.02 with premise *"tested positive for cocaine"*, `anesthesia`→0.025 with *"we'll use anesthesia"*. Those low scores are NLI answering correctly; the query is wrong, so AUC is a structural 0.677 regardless of τ. This is why **v3 uses the well-posed substitution frame** (does the original sentence entail the sentence with the span replaced by "a &lt;type&gt;" — exactly `cloak.lattice.nli_gate`). The fix worked *directionally* — AUC rose 0.677→**0.733**, the best of any method — confirming the assertion frame was mis-posed, but 0.733 is still well short of usable: at ≤10% real-loss it drops 0% junk, and the best balanced point costs 39% of real entities. So the well-posed query climbs the same wall from a slightly better foothold. skweak weak-supervision aggregation was measured and lost to the fixed layer order because its Naive-Bayes conditional-independence assumption is violated by correlated labeling functions (link and margin both read the profile).

**Conclusion: the entanglement is fundamental to the cheap-feature regime, not an artifact of any one query.** Four independent signal families — detector confidence, embedding-space distance, categorical entailment, contextual-substitution entailment — all top out AUC 0.63–0.73. The boundary cases (anatomy-as-injury, finding-as-condition, arbitrary nouns detected with high confidence) are genuinely ambiguous to every cheap feature. No post-hoc filter clears a privacy-safe operating point.

Empirical-honesty consequence (per the project's hard rule): the entanglement is the finding. The pragmatic floor is deny-list + exact link (zero false drops); the margin layer should be disabled as measured-unsafe; the real fix is upstream (a fine-tuned/higher-precision detector — the FT-detector track), not a post-hoc filter. NLI cost is not the blocker (v2 self-type is 0.08–0.12× the detector); query design is.

## Cost note

NLI viability is set by hypotheses/span: 1 hyp = 0.4× detector (0.08× on the link/deny-list residue), 3 hyps = 1.3×, 11 hyps = 4.8×. Binary self-type (or substitution) verification is practical; multi-type argmax re-routing is runtime-prohibitive and belongs offline in the miner.

## Connections

- Motivating issue: `docs/issues/2026-07-10-detector-junk-and-noise-gate-limits.md`
- Design under test: `docs/specs/detector-noise-semantic-gate.md` (layered gate — margin layer now measured-unsafe)
- Literature: SapBERT [arXiv 2010.11784], deep kNN-OOD [arXiv 2204.06507], Mahalanobis [arXiv 1807.03888], ZS4IE [arXiv 2203.13602], Sainz label-verbalization+entailment [arXiv 2109.03659], SATORI triple validation [arXiv 2401.16293], SummaC [arXiv 2111.09525], GPT-NER self-verification [arXiv 2304.10428], skweak [arXiv 2104.09683]
- Related: the substitutor's own NLI certifier (`src/cloak/lattice.py::nli_gate`) and its measured generic-level over-certification limitation (`docs/specs/substitutor-profile-match-retrieve-verify.md`)
