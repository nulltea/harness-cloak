---
type: reference
status: current
created: 2026-07-07
updated: 2026-07-07
tags: [benchmark, roundtrip, privacy, utility, detector, substitutor, extractor, re-identification]
companion: [docs/specs/detector-model.md, docs/specs/lattice-substitutor.md, docs/specs/probes.md, docs/specs/RL/roundtrip-ranker-infiller.md, docs/research/benchmarks.md, docs/specs/attacks.md]
---

# Roundtrip pipeline benchmark

## Purpose

This benchmark quantifies the full closed-box inference pipeline:

```text
doc_orig -> detector + substitutor -> doc_p -> RemoteLLM -> out_p -> extractor -> out_final
```

The headline result is a privacy-utility frontier: utility measured on `out_final`, privacy measured
against attackers on `doc_p` and leak-through from `out_final`, compared only at matched realized privacy.
Detector, substitutor, remote-task, and extractor metrics are diagnostics that explain the frontier. They are
not replacement objectives and are not normalized per method.

The benchmark is task-driven. A candidate system must preserve enough task-relevant context in `doc_p` for the
remote model to solve the user's task, while hiding direct and quasi-identifying information from the remote
model. It must then restore only the sensitive content that the remote output actually echoed or transformed
into `out_p`, so the user receives a useful `out_final` without exposing `doc_orig` upstream.

## Design Choice

I considered three benchmark shapes:

| Shape | Strength | Failure mode |
|---|---|---|
| Detector-first benchmark | Clean span metrics and broad corpus coverage | Cannot price placeholder-vs-generalization choices or extractor recovery |
| Single end-to-end leaderboard | Easy to compare methods | Hides whether a result is capped by detection misses, remote omission, substitution, or extraction |
| Decomposed task-driven benchmark | Gives stage-local diagnostics and honest end-to-end claims | More moving parts; requires pinned artifacts and cache coherence |

Use the decomposed task-driven benchmark. It is the only shape that can test the non-trivial point: replacing
everything with placeholders can improve surface privacy while damaging the task, and a truthful
generalization can improve utility while widening re-identification risk. The benchmark therefore reports
stage metrics, but accepts method claims only through the final matched-privacy frontier.

## Definitions

- **Sensitive span** - a direct or quasi-PII span in `doc_orig` that the detector should mark.
- **Task-relevant sensitive span** - a sensitive span whose information is needed by the requested task.
- **Gold-restated sensitive span** - a sensitive span whose original value, paraphrase, or task-normalized
  value is present in the reference output. This is the main utility probe supply.
- **Echoed span** - a replacement from `R` that appears in `out_p` exactly, fuzzily, or as a typed semantic
  variant. Only echoed spans give the extractor a concrete surface to invert.
- **Absorbed span** - a replacement whose information may have affected `out_p`, but whose surface leaves no
  invertible trace. Absorption can be legitimate task omission or mechanism-caused utility loss; the benchmark
  must distinguish these.
- **Realized privacy** - attacker failure on `doc_p`, plus leak-through failure on `out_final`, under the
  pre-registered attacker suite. Realized privacy is an outcome, not a proxy such as overlap, embedding
  distance, anonymity-set size, or epsilon.
- **Operating point** - a legal privacy setting of the method, such as per-type anonymity floors for the
  lattice substitutor. A learned policy may choose among legal actions, but does not get an extra privacy
  reward or per-method calibration knob.

## Benchmark Objects

Every evaluated item is a JSONL record with these fields:

```json
{
  "item_id": "clinical/mts/000123",
  "domain": "clinical",
  "task": "visit_note_generation",
  "doc_orig": "...",
  "task_prompt_template": "write_note_v1",
  "reference_outputs": ["..."],
  "gold_sensitive_spans": [
    {
      "span_id": "s17",
      "surface": "52-year-old",
      "start": 144,
      "end": 155,
      "type": "age",
      "identifier_class": "QUASI",
      "subject_id": "patient",
      "task_relevance": "gold_restated",
      "reference_evidence": ["52-year-old female"]
    }
  ],
  "privacy_targets": [
    {
      "target_id": "patient",
      "known_to_attacker": "document_context_only",
      "secret_attributes": ["name", "age", "city", "condition", "employer"]
    }
  ]
}
```

Evaluation outputs add `detected_spans`, `R`, `doc_p`, `out_p`, `out_final`, task scores, attacker traces,
and extractor traces. Artifacts are content-addressed by the tuple:

```text
benchmark_version, item_id, task_template, detector_version, substitutor_version,
privacy_setting, remote_model, extractor_version, attacker_version
```

Changing any tuple member invalidates cached claims for that cell.

## Corpora And Task Mix

### Primary utility tasks

Use tasks whose gold outputs restate or transform sensitive spans. This is mandatory because a summary or
subject line that omits all sensitive facts cannot test extraction and makes coarsening look free.

| Domain | Task | Role in benchmark | Main metrics |
|---|---|---|---|
| Clinical | dialogue to visit note on ACI-Bench and MTS-Dialog | Primary quasi-heavy task: age, dates, conditions, medications, dosage, family context | fact recall, entity-F1, section-aware note score, ROUGE-L/BERTScore as secondary |
| Legal | Multi-LexSum long case material to short case summary | Primary legal task: parties, claims, court, dates, outcomes, institutional context | fact recall, party/court/outcome F1, ROUGE-L/BERTScore as secondary |
| Biography | WikiBio biography summarization | Primary biography task: names, dates, professions, nationality/location, affiliations | fact recall, entity-F1, biography fact consistency |
| Email controls | AESLC subject generation and Enron reply generation | Negative or weak controls: useful for failure analysis, not a primary reward corpus | restatement yield, ceiling rejection, absorption decomposition |

Email was re-checked against local evidence and should not be a primary utility domain in the current
benchmark release. The earlier "email" finding is more nuanced than "all email is useless": AESLC subject
generation is clearly restatement-light (`n=3`, `gen_absent=5`, no inversion), while Enron reply generation
does fire occasionally (`n=4`, `ph_swapped=3`, `gen_exact=1`, `gen_absent=11`). Later round-trip pilot work
found both Enron and AESLC too weak for reward training because the pinned remote model produced short
pleasantries and failed the ceiling-yield gate. Keep them as documented negative controls and re-admit only
if a redesigned email task passes the same kept-facts gate as clinical, legal, and biography.

Clinical, Multi-LexSum, and WikiBio are the release-one primary utility suite. Local artifacts support that
choice: `ranker_env_full` uses clinical 267 + lexsum 161 + wikibio 160, the QA-build dev log records WikiBio
as viable while dropping QMSum as a summary desert, and the support scan passed on this mix.

### Final Dataset Decisions

Sample inspection finalized the dataset list below. The benchmark deliberately keeps the headline utility
frontier small: only datasets whose task output naturally restates or transforms sensitive facts belong in
the primary end-to-end utility suite. Other PII-rich datasets are still used, but in detector, privacy, or
smoke-test tiers where their structure actually matches the measurement.

| Dataset | Use in benchmark | Sample evidence | Decision |
|---|---|---|---|
| ACI-Bench + MTS-Dialog clinical notes | Primary end-to-end utility | Local rows contain dialogue with names, age, sex, conditions, medications, dates, and reference notes that restate many of them. Example ACI rows include "Martha Collins is a 50-year-old female..." in the note; MTS rows restate age, race, conditions, and event dates. | **Use in headline utility frontier.** |
| Multi-LexSum | Primary end-to-end utility | Local rows restate legal parties, courts, dates, statutes, claims, outcomes, and institutional context. The restatement proxy on 161 docs has mean capitalized-name carryover 0.327, enough for a legal utility slice but still worth per-item ceiling filtering. | **Use in headline utility frontier.** |
| WikiBio | Primary end-to-end utility | Local rows are dense with names, birth/death dates, places, nationality, professions, organizations, and accomplishments. The gold one-line identity summary restates the highest-value biography facts. | **Use in headline utility frontier.** |
| PriMock57 | Public clinical smoke task | Downloaded samples contain paired TextGrid doctor/patient transcripts plus JSON clinical notes. A sample note restates relative dates, symptoms, occupation, family context, social history, impression, and plan. It is only 57 consultations. | **Use as public clinical smoke / CI corpus, not headline scale.** |
| TAB legal anonymization | Detector/privacy span benchmark | Already vendored locally; provides legal direct/quasi span annotations and anonymization-oriented boundaries. | **Use in detector and privacy span evaluation.** |
| PIIBench downloadable corpus | Broad detector benchmark | Sampled Hugging Face rows are BIO token-classification records from `few_nerd`, `wikiann`, `ai4privacy_*`, and other sources. They include mixed general NER rows plus direct PII formats such as emails, driver licenses, addresses, and financial entities. | **Use for detector pretraining/eval and domain-transfer diagnostics, with source-stratified reporting. Not a utility task.** |
| Synthetic financial PII documents | Finance detector/task-construction benchmark | Downloaded Mendeley `Testing_Set.xlsx` samples include long audit/tax/compliance texts with exact character-span labels for names, emails, phone numbers, addresses, companies, URLs, SSNs, and credit cards. Some text has encoding artifacts, so normalization is required. | **Use for detector stress and a later controlled finance extraction/summarization task. Not headline until a paired task builder exists.** |
| RAT-Bench | Privacy attacker / stress benchmark | Sampled rows contain synthetic transcripts with `direct_identifiers`, `indirect_identifiers`, `features`, scenario, difficulty, and generated text. Examples combine email/name/credit card with quasi identifiers such as DOB, citizenship, state, sex, marital status, and race. | **Use for re-identification and attribute-inference stress tests. Not a utility task.** |
| AESLC + Enron | Negative/weak controls | Local samples show AESLC subject lines usually omit sensitive facts; Enron replies sometimes restate scheduling/identity facts but local remote runs produced short weak outputs. | **Keep as negative controls only.** |

Do **not** include the following in release-one benchmark runs:

| Dataset | Reason |
|---|---|
| Discharge Me! | Real clinical task fit is strong, but PhysioNet credentialed data terms and DUA restrictions make it unsuitable for the default closed-remote-LLM benchmark. Use only in a separate local/offline compliant tier. |
| ProbSum | Same PhysioNet credentialed-access issue as Discharge Me; also weaker for direct identifiers. |
| CliniKnote | Paper describes a promising 1,200-example synthetic clinical note dataset, but no public artifact/license was found. |
| MILDSum | GitHub exposes only samples; full dataset is by author contact and no visible repository license was found. |
| BillSum | Available and useful as generic summarization, but sample/task structure is weak for personal PII; it can mask regressions by making privacy look free. |
| PII-Bench query-aware paper dataset | Conceptually important for query-relevance labels, but no public artifact/license was found. Use the paper as design evidence, not as imported data. |
| QMSum | Local build exists, but the QA-build dev log and training record marked it as a summary desert with high ceiling rejection. |

### Detector coverage tasks

The detector suite is broader than the roundtrip utility suite:

- TAB legal documents for direct and quasi identifiers, including span boundary and co-reference structure.
- PIIBench downloadable corpus for broad BIO token-classification coverage and domain transfer, reported by
  source because sampled rows mix conventional NER and explicit PII.
- Synthetic financial PII documents for exact-span finance stress, including names, emails, phone numbers,
  addresses, companies, URLs, SSNs, and payment-card fields.
- Query-aware PII-Bench paper scenarios as label-design guidance only: the detector should still find all
  sensitive spans, while the benchmark separately labels which spans are relevant to the user query.
- Synthetic user-defined type slices for open-label extensibility, including gazetteer-only, zero-shot label,
  and fine-tuned type paths.

The detector benchmark reports fixed-schema recall, open-label recall, typed/boundary accuracy, and task
relevance labels separately. It must not train the detector to omit "irrelevant" sensitive spans; omission
would become a privacy leak if a future task changes.

### Privacy attack tasks

The privacy suite includes:

- `doc_p` re-identification: infer the protected person or linkable record from the anonymized prompt.
- `doc_p` attribute inference: infer original sensitive attributes or the most specific original value.
- `out_final` leak-through: test whether the returned output reveals original PII to an external party.
- Reconstruction challenge: recover original spans from `doc_p`, `out_p`, or both, depending on the attacker
  setting.

RAT-Bench is the default controlled synthetic stress corpus for direct/indirect identifier combinatorics.
Agentic-web re-identification, as in AURA, is a separate high-strength privacy tier because it uses external
web search and can dominate closed-world attackers. It should be reported separately, not mixed into the
default privacy number.

## Pipeline Evaluation

### Detector stage

The detector is scored before substitution and again as an end-to-end ceiling.

Required metrics:

| Metric | Meaning | Claim use |
|---|---|---|
| direct recall | Fraction of DIRECT spans overlapped by any predicted span | Hard privacy ceiling |
| quasi recall | Fraction of QUASI spans overlapped by any predicted span | Hard privacy ceiling |
| typed recall | Gold overlap with correct runtime type | Substitution routing quality |
| boundary recall | Gold span fully covered by prediction, not just touched | Leak residue risk |
| false-positive rate by type | Non-sensitive or wrong-subject spans routed to substitution | Utility and extractor burden |
| open-label recall | Recall on user-defined types outside the fixed schema | Tailorability |
| task relevance calibration | Ability to label, not omit, query-relevant vs query-irrelevant PII | Substitutor feature only |

Detector misses are carried into the final report as a privacy ceiling: if a sensitive span remains in
`doc_p`, the substitutor and extractor cannot fix it. The benchmark must report `doc_p` residual PII by type
and by identifier class.

### Substitutor stage

The substitutor is scored as an action allocator over detected spans. It may choose placeholders or truthful
generalizations subject to the legality mask for the privacy setting.

Required diagnostics:

| Metric | Meaning |
|---|---|
| action distribution by type | Keep, generalized text levels, typed placeholders, suppression if supported |
| legality violations | Any action outside the pre-registered privacy floor or type policy |
| task-relevance specificity | Specificity retained on gold-restated spans vs task-irrelevant spans |
| placeholder overuse penalty | Utility gap between all-placeholder and selected policy at matched privacy |
| generalization leakage | Attacker success attributable to retained generalized attributes |
| grammaticality / task parseability | Whether `doc_p` remains usable by the remote task prompt |
| injectivity and restoration safety | Whether `R` has enough information for unambiguous local restoration |

The decisive substitutor test is the counterfactual action sweep. For a fixed detector output and fixed remote
model, evaluate a small action menu per document: no privacy, all typed placeholders, coarsest legal text,
floor-walk, current policy, and learned policy. This shows whether the policy improves utility over
placeholder-only behavior at the same realized privacy, and whether it leaks more than the legal floor was
expected to permit.

### Remote task stage

Remote model behavior is part of the benchmark environment, not part of the private method. Each benchmark
release pins:

- remote model ID and provider or local serving path;
- decoding parameters, especially temperature and max tokens;
- task prompt template;
- instruction about preserving marked wording, if used;
- cache key schema.

The remote stage reports:

| Metric | Meaning |
|---|---|
| no-privacy task ceiling | Task score from `doc_orig` with no substitution |
| private pre-extraction utility | Task score on `out_p` before local restoration |
| echo rate | Replacement surfaces from `R` reproduced in `out_p` |
| gold-restated echo rate | Echo rate restricted to spans the reference output needs |
| mechanism-caused omission | Gold-restated spans present in no-privacy output but absent after substitution |
| task omission | Spans absent from both reference/no-privacy output and private output |

This split prevents raw absorption from being misread. If a task never needs a span, absorption is harmless.
If the no-privacy output uses a fact and the private output drops it, the mechanism harmed utility.

### Extractor stage

The extractor is evaluated only on information available locally: `out_p`, `R`, and the deployed extraction
model or rules. It must never call the remote model or use gold labels at inference time.

Required metrics:

| Metric | Meaning | Failure interpretation |
|---|---|---|
| placeholder restoration precision/recall | Exact placeholder echoes restored to original surfaces | Basic invertibility |
| generalized-span restoration precision/recall | Text generalizations narrowed back to originals | Main extractor competence |
| echoed-span recovery | Recovery conditioned on an invertible trace being present in `out_p` | Extractor quality isolated from remote omission |
| false substitution rate | Original PII inserted where the remote answer did not support it | Severe utility and safety failure |
| unsupported insertion count | Any original surface introduced outside an echoed/grounded site | Must be zero or guarded fallback |
| type-safe recovery | Restored surface type matches the original `R` type and local context | Prevents wrong-slot replacement |
| out_final utility delta | Utility gain from `out_p` to `out_final` | End-user value of extraction |
| leak-through delta | Attacker success increase from `out_p` to `out_final` | External-sharing risk |

Extractor claims are made at identical upstream settings. A stronger extractor has zero additional privacy
cost relative to `doc_p` under post-processing immunity for the remote threat model, but it can increase
`out_final` leak-through if the final answer is shared. The benchmark reports both facts separately.

## Utility Metrics

Headline utility is fact recall on `out_final` over gold-restated sensitive and task facts. Each probe is a
question or checkable assertion tied to a reference-supported fact. The score is macro-averaged by document,
then reported per domain; cross-domain averages are secondary.

Utility metric stack:

1. Fact recall on gold-restated sensitive spans.
2. Entity-F1 over task-relevant entities in `out_final`.
3. Task-specific factuality: clinical section facts, medication/dose/date facts, email reply intent facts.
4. ROUGE-L and BERTScore for continuity with generation literature, reported as secondary because they can
   miss exact PII restoration and overvalue fluent omissions.
5. Human or LLM-judge preference only as an audit layer, never as the sole headline.

For every utility result, report:

- no-privacy ceiling;
- `out_p` utility;
- `out_final` utility;
- all-placeholder baseline;
- floor-walk or deterministic legal baseline;
- learned policy result;
- confidence interval from document bootstrap.

## Privacy Metrics

Privacy is measured by attackers, not surface proxies.

Default attacker suite:

| Attacker | Input | Output | Score |
|---|---|---|---|
| closed-world re-identifier | `doc_p` plus candidate roster | target identity rank/top-k hit | attack success rate, MRR |
| open-world attribute attacker | `doc_p` | original sensitive attributes | exact and generalized match |
| reconstruction attacker | `doc_p` and optionally `out_p` | original span values | span recovery rate |
| leak-through attacker | `out_final` | target identity or original sensitive attributes | external-sharing leak score |

High-strength optional tier:

| Attacker | Input | Output | Score |
|---|---|---|---|
| agentic web-search re-identifier | `doc_p` with web access | person or record linkage evidence | top-k hit, evidence quality |
| frontier reasoning attacker | `doc_p` or `out_final` | identity, attributes, reconstruction | same metrics, separate budget |

The high-strength tier requires explicit approval when it costs money or uses rate-limited APIs. Local
development can use a weaker attacker, but publication claims use the pre-registered frontier attacker or
state that the frontier tier was not run.

Matched-privacy comparison rule:

1. Evaluate each method at its legitimate operating points.
2. Compute realized attacker success for every method and point.
3. Bin or interpolate by realized privacy, with pre-registered bin widths.
4. Compare utility only within matched realized privacy bins.
5. If bins have too little overlap, report non-overlap rather than normalizing a secondary quantity.

## Benchmark Baselines

Every benchmark release includes these controls:

| Baseline | Purpose |
|---|---|
| no privacy | Remote task ceiling and restatement supply |
| oracle detector + current substitutor | Detector miss ceiling |
| current detector + oracle-safe span list | Detector false-positive burden and routing burden |
| all typed placeholders | Tests the "replace everything" strategy |
| coarsest legal text | Tests maximal generalization without placeholders |
| deterministic floor-walk | Strong local legal baseline |
| current deployed policy | Regression guard |
| learned policy | Candidate RL or ranker policy |
| extractor off | Value of `out_final` restoration |
| oracle extractor on echoed spans | Extractor headroom given remote echo |

External systems such as RUPTA or Staab-style adversarial anonymizers may be run only if they can be adapted
to the same input/output contract and evaluated with the same remote task, extractor-off condition, and
attacker suite. If they rewrite full text without an `R` record, they are end-to-end baselines, not drop-in
substitutors.

## Slices And Reporting

All headline tables are stratified by:

- domain: clinical, email, legal, social/synthetic;
- identifier class: DIRECT, QUASI;
- runtime type: PERSON, CODE, LOC, ORG, DATETIME, QUANTITY, MISC, and fine demographic leaves;
- task relevance: gold-restated, task-relevant but not restated, task-irrelevant;
- subject structure: single subject, multi-subject, sibling/relative mention, organization-mediated identity;
- substitution action: placeholder, textual generalization depth, keep-if-legal;
- remote behavior: exact echo, fuzzy echo, semantic echo, absorbed, omitted.

Do not collapse these strata before inspecting regressions. A method that improves average utility by
restoring clinical ages while leaking sibling mentions or rare organizations has not earned a clean win.

## Acceptance Gates

A benchmarked candidate can be called better only if all are true:

- Detector residual leak does not increase on DIRECT or QUASI spans at the selected detector operating point.
- No legality violations in `R`.
- No unsupported extractor insertion above the pre-registered tolerance; default tolerance is zero.
- Utility on `out_final` improves over the deterministic legal baseline at matched realized privacy, or is
  statistically equivalent with a clearly lower cost.
- Privacy on `doc_p` is measured by the attacker suite, not inferred from anonymity floors.
- `out_final` leak-through is reported, even when the remote-threat privacy claim is only about `doc_p`.
- Results are reported per domain and privacy bin, with document-bootstrap confidence intervals.
- Degenerate outcomes are reported as findings: all-placeholder collapse, no echo, detector miss ceiling,
  or attacker breakage must not be engineered away after seeing the result.

## Implementation Notes

The benchmark should be implemented as a harness, not a monolithic script:

| Component | Responsibility |
|---|---|
| corpus registry | Loads item JSONL records and reference outputs |
| detector runner | Produces detected spans and residual-leak audit |
| substitution runner | Produces `doc_p` and `R` for each operating point |
| remote runner | Executes pinned task prompts with content-addressed caching |
| extraction runner | Produces `out_final` and trace-level recovery labels |
| utility scorer | Computes fact recall, entity-F1, and generation metrics |
| attacker runner | Computes realized privacy on `doc_p` and leak-through on `out_final` |
| reporter | Builds per-domain tables, Pareto curves, and regression summaries |

Before a heavy run, the harness must pass the project performance gate: smallest slice that answers the
question, batched local inference where possible, remote calls cached, and one GPU process at a time.

Release-one implementation entry points:

- `src/bench/` owns schema, suite registry, baseline policies, runner, metrics, deterministic privacy
  attackers, and report generation.
- `scripts/run_roundtrip_benchmark.py` is the durable CLI. `--dry-run` writes only manifest/items;
  `--stub-remote` exercises the full local trace/scoring/report path without remote cost.
- Live remote runs require `INFERDPT_LLM_CACHE` before the remote client is constructed.
- Runs must pass explicit model arguments for every model-bearing stage so the manifest and config hash carry
  the full benchmark identity. Current detector runs require `--detector-model`; the release-one fine-dem
  detector command uses `--detector-model data/models/pii_gliner_finedem/final --detector-fine-dem`.
  Attacker model-bearing tiers must pass `--attack-docp-model`, `--attack-reconstruction-model`, and
  `--attack-leak-model`; the deterministic `offline-v1` tier may record these as `offline-*` identifiers.

## Literature Grounding

Sources contributed: local project docs/research-wiki and primary web pages. Local `papers/` and
`literature/` had no PDFs. ARIS paper-verification helpers were not present, so new external candidates were
verified by opening primary arXiv or ACL pages directly.

| Source | Verification status | Benchmark implication |
|---|---|---|
| [`pilan2022_tab_benchmark`](../../research-wiki/papers/pilan2022_tab_benchmark.md) ([arXiv 2202.00443](https://arxiv.org/abs/2202.00443), [DOI 10.1162/coli_a_00458](https://doi.org/10.1162/coli_a_00458)) | verified via local wiki + arXiv | Direct/quasi span detection and privacy-oriented span evaluation |
| [`yim2023_acibench_visit_note_generation`](../../research-wiki/papers/yim2023_acibench_visit_note_generation.md) ([arXiv 2306.02022](https://arxiv.org/abs/2306.02022), [DOI 10.1038/s41597-023-02487-3](https://doi.org/10.1038/s41597-023-02487-3)) | verified via local wiki | Clinical note utility task with quasi restatement |
| [`benabacha2023_mtsdialog_clinical_note`](../../research-wiki/papers/benabacha2023_mtsdialog_clinical_note.md) ([ACL 2023.eacl-main.168](https://aclanthology.org/2023.eacl-main.168/)) | verified via local wiki | Larger clinical note utility task |
| [`zhang2019_aeslc_subject_line_generation`](../../research-wiki/papers/zhang2019_aeslc_subject_line_generation.md) ([arXiv 1906.03497](https://arxiv.org/abs/1906.03497), [ACL P19-1043](https://aclanthology.org/P19-1043/)) | verified via local wiki | Email source; local runs demote AESLC/Enron to weak-control status |
| Multi-LexSum ([project page](https://multilexsum.github.io/), [arXiv 2206.10883](https://arxiv.org/abs/2206.10883)) | verified via project page | Primary legal case-summary task; expert summaries at multiple granularities |
| WikiBio ([GitHub](https://github.com/DavidGrangier/wikipedia-biography-dataset), [arXiv 1603.07771](https://arxiv.org/abs/1603.07771)) | verified via GitHub | Primary biography task; dense names/dates/professions/locations |
| Discharge Me! ([project page](https://stanford-aimi.github.io/discharge-me/), [PhysioNet](https://physionet.org/content/discharge-me/1.2/)) | verified via project page + PhysioNet; credentialed DUA, PhysioNet Credentialed Health Data License 1.5.0 | Candidate clinical EHR expansion; DUA-gated and not default remote-API-safe |
| ProbSum ([arXiv 2306.05270](https://arxiv.org/abs/2306.05270), [PhysioNet](https://physionet.org/content/bionlp-workshop-2023-task-1a/2.0.0/)) | verified via arXiv + PhysioNet; credentialed DUA, PhysioNet Credentialed Health Data License 1.5.0 | Candidate clinical problem-list generation task, local/offline only unless compliance changes |
| PriMock57 ([ACL 2022](https://aclanthology.org/2022.acl-short.65/), [GitHub](https://github.com/babylonhealth/primock57), [DOI 10.18653/v1/2022.acl-short.65](https://doi.org/10.18653/v1/2022.acl-short.65)) | verified via ACL Anthology + GitHub; CC BY 4.0; sampled via downloader | Public mock-consultation smoke task with paired transcript/note data |
| CliniKnote ([arXiv 2408.14568](https://arxiv.org/abs/2408.14568)) | paper verified via arXiv; no public artifact/license found | Hold as unreleased candidate synthetic/curated clinical note expansion |
| MILDSum ([arXiv 2310.18600](https://arxiv.org/abs/2310.18600), [GitHub](https://github.com/Law-AI/MILDSum)) | verified via arXiv + GitHub; samples public, full dataset by contact, no visible repository license | Candidate legal summarization expansion after access/license approval |
| BillSum ([arXiv 1910.00523](https://arxiv.org/abs/1910.00523), [Hugging Face](https://huggingface.co/datasets/FiscalNote/billsum)) | verified via arXiv + Hugging Face; `cc0-1.0` tag with US GPO Govinfo source under CC0-1.0 | Non-PII legal summarization control, not primary |
| [`staab2024_llm_anonymizers`](../../research-wiki/papers/staab2024_llm_anonymizers.md) ([arXiv 2402.13846](https://arxiv.org/abs/2402.13846)) | verified via local wiki + arXiv | LLM adversary should define privacy, not surface metrics |
| [`yang2025_rupta`](../../research-wiki/papers/yang2025_rupta.md) ([arXiv 2407.11770](https://arxiv.org/abs/2407.11770)) | verified via local wiki + arXiv | Privacy-utility optimization under LLM re-identification, with circularity caution |
| [`pang2024_reconstruction_dp_text_llm`](../../research-wiki/papers/pang2024_reconstruction_dp_text_llm.md) ([arXiv 2410.12443](https://arxiv.org/abs/2410.12443)) | verified via local wiki + arXiv | Reconstruction attacker against sanitized text |
| [`tong2025_vulnerability_text_sanitization`](../../research-wiki/papers/tong2025_vulnerability_text_sanitization.md) ([arXiv 2410.17052](https://arxiv.org/abs/2410.17052)) | verified via local wiki + arXiv | Optimal/practical reconstruction attacks as privacy stress tests |
| [`jha2026_piibench_deberta`](../../research-wiki/papers/jha2026_piibench_deberta.md) ([arXiv 2605.25816](https://arxiv.org/abs/2605.25816)) and PIIBench corpus ([arXiv 2604.15776](https://arxiv.org/abs/2604.15776)) | local wiki + arXiv | Broad PII detection and domain-transfer stress |
| PII-Bench query-aware privacy systems ([ACL 2026](https://aclanthology.org/2026.acl-long.227/), [arXiv 2502.18545](https://arxiv.org/abs/2502.18545), [DOI 10.18653/v1/2026.acl-long.227](https://doi.org/10.18653/v1/2026.acl-long.227)) | verified via ACL Anthology + arXiv; no public artifact/license found | Separate "detect all PII" from "choose task-relevant masking/generalization" |
| PIIBench downloadable corpus ([Hugging Face](https://huggingface.co/datasets/Pritesh-2711/pii-bench), [arXiv 2604.15776](https://arxiv.org/abs/2604.15776)) | verified via Hugging Face; Apache-2.0; sampled `test.jsonl` | Downloadable broad BIO PII detector corpus; large, guarded downloader |
| RAT-Bench ([arXiv 2602.12806](https://arxiv.org/abs/2602.12806), [Hugging Face](https://huggingface.co/datasets/imperial-cpg/rat-bench)) | verified via arXiv + Hugging Face; MIT license; sampled English level 1 | Controlled direct/indirect identifier re-identification and attribute-inference risk |
| Synthetic financial PII dataset ([Mendeley Data DOI 10.17632/tzrjx692jy.1](https://doi.org/10.17632/tzrjx692jy.1)) | verified via Mendeley Data; CC BY 4.0; sampled `Testing_Set.xlsx` | Synthetic finance detector stress and later task-construction source |
| AURA ([arXiv 2605.30848](https://arxiv.org/abs/2605.30848)) | verified via arXiv | Optional high-strength agentic web-search privacy tier |
| LLM-PBE ([arXiv 2408.12787](https://arxiv.org/abs/2408.12787)) | verified via arXiv | Privacy attack harness ideas and Enron/ECHR/PubMed attack coverage |
| Privacy Evaluation Benchmarks for NLP Models ([arXiv 2409.15868](https://arxiv.org/abs/2409.15868)) | verified via arXiv | Standardized attack/defense protocol framing |

## Open Decisions

- Whether agentic web-search re-identification is part of the default benchmark or a paid/high-strength tier.
  Recommendation: high-strength tier only.
- Whether reference-scored generation metrics should ever be gates. Recommendation: no; use them as secondary
  continuity metrics behind fact recall and entity-F1.
- Which remote LLM is the default task executor for benchmark release one. Recommendation: reuse the pinned
  roundtrip RL environment unless the benchmark release deliberately re-pins and invalidates old caches.
- Whether to promote the synthetic financial documents from detector stress to an end-to-end utility task.
  Recommendation: only after a paired form-extraction or statement-summary task builder passes a no-privacy
  ceiling run.
- Whether to register the newly adopted public datasets in `research-wiki/papers/`. Recommendation: register
  PriMock57, RAT-Bench, PIIBench, and the synthetic financial PII dataset when their local corpus builders
  are added, so the wiki tracks adopted evidence rather than a wish list.
