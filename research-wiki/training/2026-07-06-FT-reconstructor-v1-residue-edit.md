---
type: training-experiment
status: planned
created: 2026-07-06
model: google/flan-t5-base (+ LoRA r=16, q/v)
dataset: reconstructor_{clinical,lexsum}.jsonl — residue-edit triples distilled from the Qwen3.6 survival judge
result: pending
tags: [reconstructor, design3, residue-edit, lora, flan-t5, survived-recovery]
companion: ../../docs/plans/2026-07-05-survived-recovery-extractor.md
---

# FT-reconstructor v1 — residue-targeted constrained edit (judge-distilled)

Spec written before the run (CLAUDE.md training-experiment schema); Results filled after.
Companion design: `docs/plans/2026-07-05-survived-recovery-extractor.md`; implementation plan:
`docs/plans/2026-07-06-reconstructor-design3-plan.md`.

## Objective & hypothesis

Recover the survived-span residue the deterministic cascade (`cloak.extract.invert`, semantic-window
default) misses — paraphrase-reworded / lossy fill mentions (D-1 semantic, D-3) — by editing `out_p`
with the original surfaces in R, pushing survived-span recovery from the cascade's ~82% toward the
client-side ceiling (~100%). **Hypothesis:** a small seq2seq (flan-t5-base + LoRA) that reads the
cascade's partial output plus a linearized restore-map, gated by the copy-bias `edit_guard`, restores
reworded residue mentions the rule cascade cannot align, at **zero false substitutions** (do-no-harm).

## Training data (sources + explicit ratios + type-mapping)

- **Source:** `scripts/build_reconstructor_data.py` distills the Qwen3.6-35B-A3B survival judge into
  `(input, target)` triples. `input = <cascade out_final'>\n\n[RESTORE]\n<linearized residue>`;
  `target` = the cascade output with each **admitted** located mention spliced to its original.
- **Admission gate** (`cloak.reconstruct.restorable`): judge label ∈ {SURVIVED, REWORDED} ∧ quote
  grounded in the edited text ∧ `_type_sane`; for ambiguous types {DATETIME, QUANTITY, LOC, ORG} a
  mandatory `fill ⊨ quote` correspondence check (deterministic `_value_compatible` fail-closed, NLI
  fallback via `cross-encoder/nli-deberta-v3-small`). Rejected-but-grounded → no-op target + logged to
  `data/reconstructor_<corpus>_degeneracies.jsonl`.
- **No-op handling** (`train_reconstructor.load`): all residue-positive edits kept; high-risk no-ops
  (D-4-like / scalar-date rejects) kept UNCAPPED as a safety signal; generic no-ops capped at
  `noop_cap=0.3` of the positive count.
- **Train/eval disjointness (open-label generality):** per Global Constraints, never average across
  corpora. Three checkpoints, three strata:
  - `reconstructor_v1` — trained on **all clinical** (`--split all`), evaluated **cross-domain** on lexsum.
  - `reconstructor_clinical_split` — trained on **clinical train-split** (`--split train`), evaluated on
    the **clinical held-out** complement.
  - `reconstructor_lexsum_split` — trained on **lexsum train-split**, evaluated on **lexsum held-out**.
  - Doc-level split = deterministic seeded hash reusing the env `split_seed` + `held_out_frac`
    (`data/recon_train_ids_<corpus>.txt`, written by the builder).
- **Ratios:** measured after the build (row counts per corpus: positive / high-risk no-op / generic
  no-op) — filled in Results.

## Training config

- Base `google/flan-t5-base`, LoRA `r=16, alpha=32, dropout=0.05, target_modules=[q,v]`, `SEQ_2_SEQ_LM`.
- `epochs=4`, `per_device_train_batch_size=4`, `lr=2e-4`, `bf16`, `save_strategy=epoch`, `max_length=1024`.
- Pinned versions verified installed: `transformers==5.12.1`, `peft==0.19.1`, torch ROCm (gfx1151).
- One GPU process at a time; `.venv/bin/python -u`, logs to `results/train_reconstructor*.log`.

## Selection & operating point

Single operating point (no sweep for v1): the epoch-4 checkpoint. Selection is by the Task-6 eval,
not train loss — recovery must beat the cascade at zero false substitutions. `edit_guard` at inference
uses `max_edits=2*len(residue)+1`.

## Evaluation & success criteria

`scripts/spikes/reconstructor_eval.py` — per-residue quote-anchored `classify_recovery`
(recovered / wrong_insert / deletion / miss), run ONCE PER STRATUM, never averaged. All must hold:
- `recon.recovered > cascade.recovered` on D-residue types, per stratum incl. ≥1 **in-domain held-out**
  (not only cross-domain);
- `recon.wrong_insert == 0` and `recon.deletion` not worse than cascade;
- `harm_rate == 0` — zero cascade-resolved spans altered by the reconstructor.
Any nonzero `harm_rate` or `wrong_insert` is a **hard fail** → tighten `edit_guard` / admission gate /
no-op weighting before claiming a win.

## Results (measured or pending)

**pending** — filled after the eval runs (per-type recovery per stratum; report the win AND any
regression / false-sub per the empirical-honesty rule).

## Ablations

Planned if v1 passes: (a) no-op cap sensitivity; (b) with/without the NLI correspondence gate (D-4
leakage rate); (c) guard `max_edits` bound. Deferred until a v1 win exists.

## Cost

Build: proxy/judge-bound (Qwen `-np 1` serial + cached roundtrips + local NLI), ~15–20 min for
clinical+lexsum n=80. Train: flan-t5-base LoRA on a few hundred short triples, a few min/epoch × 4;
3 checkpoints. Eval: 3 strata, judge calls (cached where warm) + local model.generate. Measured cost
filled in Results.

## Risks & caveats

- `edit_guard` is word-level diff + fuzzy-anchor (a deliberate ceiling); it bounds inserted content to
  R originals but permits bounded deletions — if `harm_rate`/`wrong_insert` is nonzero, upgrade to
  token-constrained decoding (`prefix_allowed_tokens_fn`) before adding capacity.
- Admitted-target precision gate: hand-audit 30 admitted targets pre-training; **do not train at
  < ~0.95 precision** (a false restoration is worse than a miss).
- The build's out_p pin (gemma E4B) + judge pin (Qwen3.6) are a re-gate surface; the semantic-window
  cascade default (commit 97c26af) is the baseline the reconstructor is measured against.

## Artifacts (paths)

- Data: `data/reconstructor_{clinical,lexsum}.jsonl`, `..._degeneracies.jsonl`,
  `data/recon_train_ids_{clinical,lexsum}.txt`.
- Checkpoints: `data/models/reconstructor_v1`, `..._clinical_split`, `..._lexsum_split`.
- Eval: `results/reconstructor_eval_{crossdomain,clinical_heldout,lexsum_heldout}.json`.
- Code: `src/cloak/reconstruct.py`, `scripts/build_reconstructor_data.py`,
  `scripts/train_reconstructor.py`, `scripts/spikes/reconstructor_eval.py`.

## Sources

- Design: `docs/plans/2026-07-05-survived-recovery-extractor.md`,
  `docs/plans/2026-07-06-reconstructor-design3-plan.md`.
- Measured ground truth: `research-wiki/experiments/extractor-pointer-by-type.md` (151-doc survival,
  293/1059 survive, ~82% recovered by the cascade).
