---
type: reference
status: current
created: 2026-07-02
updated: 2026-07-02
tags: [benchmarks, eval, task-utility, pii-rich, clinical-note, email, pareto, spec]
companion: docs/specs/probes.md
---

# Benchmark specifications — task-oriented, PII-rich evaluation

Living doc. Defines the corpora + tasks LatticeCloak is evaluated on. Updated as the eval harness lands.

## Why (the §5.4 gap)

The smoke eval used prefix-continuation / summarization / attribute-QA on SynthPAI. Those answers
**paraphrase around** the substituted spans (128/128 substituted spans absent from the 32 answers), so:

- **rule-based inversion never fires** — nothing to narrow back, extraction untested;
- **utility is insensitive to τ** — coarsening a span the answer never restates costs nothing measurable.

Neither the extraction hypothesis (rule extractor vs learned extractor) nor the τ→utility Pareto can be priced on such tasks.

## Selection criterion

Adopt tasks whose **gold output reproduces or transforms the sensitive spans**. Two properties, both required:

1. **PII-rich input** — spans to detect/substitute.
2. **Output-restates-input** — the reference answer legitimately contains the entities, so inversion is
   exercised and coarsening carries a reference-scored utility cost.

SynthPAI has (1), not (2). Generation / rewriting / note-writing / reply-drafting have both.

## Adopted benchmarks (two domains)

### Clinical — dialogue → note (primary, full quasi-identifier load)

| corpus | size | reference | PII / quasi content |
|---|---|---|---|
| ACI-Bench | 207 encounters | full SOAP notes | age, sex, name (meta), conditions, meds, dosages, dates |
| MTS-Dialog | 1,700 dialogues | section-header notes | same, concise single-section |

- Wiki: [`yim2023_acibench_visit_note_generation`](../../research-wiki/papers/yim2023_acibench_visit_note_generation.md) ([arXiv 2306.02022](https://arxiv.org/abs/2306.02022) · [DOI 10.1038/s41597-023-02487-3](https://doi.org/10.1038/s41597-023-02487-3)); [`benabacha2023_mtsdialog_clinical_note`](../../research-wiki/papers/benabacha2023_mtsdialog_clinical_note.md) ([ACL 2023.eacl-main.168](https://aclanthology.org/2023.eacl-main.168/)).
- Task: generate the visit note from the dialogue. The note **must** restate the coarsened quasi-identifiers → inversion fires, utility sensitive to τ.
- Caveat: gold notes are lightly de-identified on **direct names** (placeholders barely tested here); the **quasi** load (age/meds/dates) is intact — that is the part lattices touch, so τ sensitivity holds. Direct-placeholder inversion is stress-tested on the email domain instead.

### Email — Enron (real PII, direct + quasi)

- Wiki: [`zhang2019_aeslc_subject_line_generation`](../../research-wiki/papers/zhang2019_aeslc_subject_line_generation.md) ([arXiv 1906.03497](https://arxiv.org/abs/1906.03497) · [ACL P19-1043](https://aclanthology.org/P19-1043/)).
- **`aeslc`** — subject-line generation (multi-reference gold, short → **light** restatement; smoke confirmed inversion barely fires).
- **`enron`** — email reply generation (fuller restatement: reply cites names/orgs/dates/amounts). CMU Enron strips threading headers, so gold pairs are mined from **quoted top-posted replies within one message**: gold = the new top text, parent = the quoted original (`build_enron` in build_task_corpora.py). Real people → real PII. This is the direct-placeholder stress test the clinical de-identified notes can't provide.

## Pipeline & metrics

Per item: `doc_orig → doc_p (+R) → remote LLM → out_p → rule extractor → out_final`.

- **Utility** — reference-scored on `out_final`: ROUGE-L, BERTScore; clinical adds a fact-extraction/entity-F1 metric (note reference standard). Report `out_p` (pre-inversion) and `out_ctrl` (no-privacy) alongside.
- **Inversion stats** — placeholders swapped, generalizations narrowed exact/fuzzy, spans absent (the §5.4 table, but now non-trivially exercised).
- **Privacy** — attacker on `doc_p` and `out_final` (separate axis, see attacker sources below). Pareto: sweep τ, plot utility vs realized privacy **at matched realized privacy** — never a per-model fudge to equalize a secondary quantity (project honesty rule).

## Attacker axis (privacy, not utility — for the Pareto's other axis)

Kept separate; do **not** conflate with the utility benchmarks above.

- SynthPAI — attribute inference (already in use), stays as a privacy corpus.
- LLM-PBE ([arXiv 2408.12787](https://arxiv.org/abs/2408.12787), VLDB'24) — bundles Enron/ECHR/PubMed extraction attacks; register + wire when the adversarial stage is built.
- Contextual-integrity/agent-leakage benchmarks (ConfAIde, PrivacyLens) — **different threat model** (agent decides what to disclose), not the closed-box rewrite setting; reference only.

## Harness

`--corpus {aci,mts,clinical,aeslc,enron}` across two scripts, both: doc → substitute(tau) → remote gen
(`out_p`, `out_ctrl`) → `extract.invert` → `out_final` → `score.score_batch` (ROUGE-L always, BERTScore
behind `--bertscore`).

- `scripts/latticecloak_task_eval.py` — single-tau eval (`--dry-run` validates load+substitute+score, no remote).
- `scripts/latticecloak_task_tau_sweep.py` — tau sweep; efficient (detect once, `out_ctrl` once, substitute
  per tau), records lattice mechanics (`at_most_specific`/`at_floor`) + inversion totals.

Corpora built by `scripts/build_task_corpora.py`; loaders `src/cloak/corpora.py` (clinical round-robins
aci+mts so a slice hits both), templates `src/cloak/tasks.py`.

## Smoke results (2026-07-02, tau=0.02, ROUGE-L)

Tiny, not a claim — validates the pipeline and the selection criterion.

- **ACI (n=2):** inversion **fires** — gen_exact=1, gen_fuzzy=1 (vs 0/0 in the old SynthPAI smoke); `out_p`
  "fifty-something female" → `out_final` "50-year-old female". utility final 0.320 / ctrl 0.282 / p 0.317.
  Clinical is the task that exercises inversion, as designed.
- **AESLC subject (n=3):** inversion **all-absent** (gen 0/0/0, utility_p == utility_final); ctrl 0.521 >
  final 0.471. Subject lines are too short to restate substituted spans → confirms the light-restatement
  caveat; the **Enron reply/body variant is needed** for real direct-placeholder coverage.
- Substitution-quality artifact surfaced on clinical: a lowercase dialogue name ("martha") routes to a
  demographic generalization ("a personal attribute") rather than a PERSON placeholder — detection/routing
  issue (§ detect retyping rule), tracked separately from the eval harness.

## Open items

- [x] Data materialized: `corpora/clinical/{aci (67), mts (200)}`, `corpora/aeslc/test (200)`, `corpora/enron/replies (200)`.
- [x] Harness + scorer wired and smoked (ACI, AESLC).
- [x] Enron reply variant built (quoted top-post pairs; AESLC subject-line load too light per smoke).
- [x] Fixed lowercase-name → demographic routing: role nouns generalize, names hit the PERSON placeholder path (WordNet discriminator, `substitute._is_role_phrase`).
- [x] Real tau-sweep on ACI/MTS with `--bertscore` (`latticecloak_task_tau_sweep.py`).
- [ ] License check before redistributing slices (microsoft clinical corpus; Enron/AESLC).
- [ ] Confirm clinical note references usable as gold without penalizing licensed omission (add entity/fact-F1).
- [ ] Ceiling: a name that is also a common noun ("Bill", "Rose") lowercased still misroutes to DEM — add a names gazetteer if it bites.
- [ ] Batch MTI probing across spans — sequential (batch=1) probing is the tau-sweep bottleneck at scale.
- [ ] The privacy Pareto (attacker axis); register LLM-PBE when that stage lands.
