---
type: research
status: current
created: 2026-07-04
updated: 2026-07-04
tags: [datasets, pii-detection, quasi-identifier, benchmark, de-identification, anonymization, tab, synthpai]
companion: docs/research/learned-PII-detection.md
---

# PII / anonymization dataset taxonomy: what has DIRECT/QUASI, what we use, what we don't

Catalogue of the datasets relevant to the cloak pipeline's span detector and round-trip evaluation,
taxonomized by the one axis that matters for us — **whether a dataset annotates quasi-identifiers**
(the re-identification-risk spans an attacker aggregates) or only formal/direct PII. Companion to the
detector survey (`learned-PII-detection.md`); that doc argues *why* QUASI is the gap, this doc records
*which corpora* carry it. Split into **datasets we use locally** and **datasets discovered** in the
2026-07-04 survey. Verified arXiv/DOI/HF identifiers throughout; items the survey could not confirm are
marked **UNVERIFIED**.

## Definitions

- **PII** — personally identifiable information; any span identifying a person directly or in combination.
- **DIRECT identifier** — identifies a person alone (name, case/SSN number).
- **QUASI identifier (quasi-identifier)** — identifies only *in combination* (date + profession + city;
  an identifying event). The aggregation target of a re-identification attacker.
- **Formal PII** — the fixed catalogue most datasets annotate: names, emails, phones, account/ID codes,
  addresses, dates, cards. Its *types* partly overlap QUASI (dates, locations) but it is not labeled by
  re-identification risk.
- **Mask-all** — a de-identification scheme that redacts every PHI/PII span without a direct-vs-quasi
  distinction (e.g. HIPAA de-id). Type inventory can be quasi-*rich* yet carry no quasi *semantics*.
- **Span-level vs attribute-level** — span: character/token offsets of the mention (what a detector
  needs). Attribute: a document/comment tagged with a person-attribute value (age=34), no span.
- **Real (naturally-occurring) vs synthetic** — real: human-written text (court cases, clinical notes,
  essays). Synthetic: machine-/template-generated. Detectors trained on synthetic transfer poorly to
  real prose (the benchmark-honesty problem, `learned-PII-detection.md` §4).
- **Surrogate substitution** — real text whose PII *values* are swapped for realistic fakes before
  release (i2b2, PIILO): the prose is real, the identifiers are not the originals.
- **Gated / DUA** — access requires credentialing + a signed Data Use Agreement (clinical corpora).
- **Span-F1 / any-recall** — detection metrics; "any" counts a gold mention detected on any character
  overlap (the gate's recall metric).

## Taxonomy — the four tiers by quasi-awareness

The primary axis is quasi-awareness; secondary axes (real/synthetic, span/attribute, train/test,
access) are in the tables. **Tier 1 is the only bucket with true DIRECT/QUASI span labels, and it has
exactly two members.**

### Tier 1 — true DIRECT/QUASI span annotation (the TAB scheme)

| Dataset | Domain | Size | Real? | Quasi annotation | Access | Role | Source |
|---|---|---|---|---|---|---|---|
| **TAB** (Text Anonymization Benchmark) | ECHR legal | 1,268 docs | ✔ real | DIRECT/QUASI, 8 entity types, NO_MASK, coref | open | **train + test (ours)** | [pilan2022_tab_benchmark](../../research-wiki/papers/pilan2022_tab_benchmark.md) ([arXiv 2202.00443](https://arxiv.org/abs/2202.00443)) |
| **Wikipedia biographies** (Papadopoulou et al.) | biographies | 553 docs (453 train / 100 test) | ✔ real | **same DIRECT/QUASI + mask-decision scheme** (same Norsk Regnesentral group); ≈14% DIRECT, 56% QUASI-masked, 30% unmasked | open (NR text-anonymization repo) | train + cross-domain test | [arXiv 2205.06895](https://arxiv.org/abs/2205.06895); reused in [arXiv 2310.14312](https://arxiv.org/abs/2310.14312) |

These are the only two corpora with TAB's direct-vs-quasi + mask-decision span labeling, both real
text, both from the same group. The Wikipedia-bio set is the one true cross-domain complement to TAB.

### Tier 2 — quasi-rich *types*, span-labeled, but mask-all (no direct/quasi split)

Useful to *reinforce* detection of QUASI-type spans (dates/ages/locations/professions), not to learn
the direct-vs-quasi distinction.

| Dataset | Domain | Size | Real? | Quasi-type coverage | Access | Role | Source |
|---|---|---|---|---|---|---|---|
| **i2b2/UTHealth 2014 de-id** (canonical clinical) | clinical notes | 1,304 records / 296 patients; 28,872 PHI (790 train / 514 test) | ✔ real (surrogate PHI) | explicit **DATE, AGE, PROFESSION, LOCATION**(→city/state/country/…), NAME, CONTACT, ID | **DUA-gated** (Harvard DBMI) | train + test (gated) | DOI [10.1016/j.jbi.2015.07.020](https://doi.org/10.1016/j.jbi.2015.07.020) |
| **2016 CEGS N-GRID** (psychiatric) | psychiatric intake | 1,000 records | ✔ real (surrogate) | i2b2-2014 schema → DATE/AGE/PROFESSION/LOCATION; hardest generalization track | DUA-gated | train + sight-unseen test | DOI [10.1016/j.jbi.2017.06.011](https://doi.org/10.1016/j.jbi.2017.06.011) |
| **i2b2 2006 de-id** | discharge summaries | 889 docs; ~19.5k PHI (UNVERIFIED) | ✔ real (surrogate) | DATE, LOCATION, AGE (no PROFESSION) | DUA-gated | train + test | DOI [10.1197/jamia.M2444](https://doi.org/10.1197/jamia.M2444) |
| **REDACT** (ServiceNow) | 12 domains, 25 langs | 13,427 records / 324,078 annotations | ✗ synthetic | **richest explicit quasi list**: nationality, place of birth, occupation/title, gender, marital status, DOB + GDPR special categories; disclosure_form + sensitivity_tier tags | open, CC-BY-4.0 (AUP) | **eval-only** | [arXiv 2606.19881](https://arxiv.org/abs/2606.19881) |
| **NVIDIA Nemotron-PII** | 50+ industries | ~100k–200k rows (UNVERIFIED) | ✗ synthetic | 55+ types incl. ages, nationalities/ethnicities, occupations, locations, dates | open, CC-BY-4.0 | train + test | [HF nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) |
| **ai4privacy pii-masking-200k** | assistant text | 200k | ✗ synthetic | DOB, DATE, TIME, AGE, SEX, CITY/STATE/STREET, GPS, JOBTITLE/JOBAREA | open (DOI 10.57967/hf/1532) | train | [HF ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) |
| **Presidio-research** (Microsoft) | Faker templates | generator (NA) | ✗ synthetic | LOCATION, DATE_TIME, **NRP** (nationality/religion/politics), URL + formal IDs | open, MIT | train + eval | [github microsoft/presidio-research](https://github.com/microsoft/presidio-research) |

### Tier 3 — attribute-inference (labels *are* quasi-identifiers, but attribute/comment-level, not spans)

For re-identification-attacker training/eval, not span detection.

| Dataset | Domain | Size | Real? | Quasi labels | Access | Role | Source |
|---|---|---|---|---|---|---|---|
| **SynthPAI** | synthetic Reddit | 7,800+ comments / 300 profiles | ✗ synthetic (human-validated) | 8 census attributes (age, education, income, location, occupation, birthplace, relationship, sex) | open (HF + github eth-sri) | **train + test (attacker); used in pipeline** | [arXiv 2406.07217](https://arxiv.org/abs/2406.07217) |
| **PersonalReddit** (Beyond Memorization) | real Reddit | 520 profiles / 5,814 comments (raw restricted; 525 synth released) | ✔ real (restricted) | same 8 attributes | raw restricted; synth public | eval | [staab2024_llm_anonymizers](../../research-wiki/papers/staab2024_llm_anonymizers.md) ([arXiv 2310.07298](https://arxiv.org/abs/2310.07298)) |
| **RAT-Bench** | medical/chat/meeting transcripts | 100 EN (+50 ES, +50 zh) | hybrid (Census-grounded) | distinguishes direct vs **indirect** identifiers, grounded in US Census PUMS; attribute-vector GT, not spans | open (HF) | eval-only | [arXiv 2602.12806](https://arxiv.org/abs/2602.12806) |
| **Personal Facts in Dialogue** | PersonaChat/MSC dialogue | UNVERIFIED | ✔ real | span-level personal-fact categories (not framed direct/quasi) | HF `adugeen/personal-facts-msc`, CC-BY-4.0 | train + test | [arXiv 2605.10339](https://arxiv.org/abs/2605.10339) |

### Tier 4 — formal-PII-only or general NER (no quasi semantics)

Auxiliary breadth for PERSON/CODE/LOC only. **Most are synthetic.**

| Dataset | Domain | Real? | Types | Quasi? | Source |
|---|---|---|---|---|---|
| **ai4privacy** (43k…500k, openpii-1.5m) | assistant text | ✗ synthetic | 17–54 formal PII | some dates/locations in mid sizes | [HF ai4privacy](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy) |
| **Kaggle PII / PIILO** | student essays | ✔ real (surrogate values) | 7 formal (NAME/EMAIL/USERNAME/ID/PHONE/URL/STREET) | ✗ none | [PIILO Kaggle](https://www.kaggle.com/datasets/lburleigh/piilo-dataset), DOI [10.1108/ILS-04-2023-0032](https://doi.org/10.1108/ILS-04-2023-0032) |
| **SPY** (Synthetic PII) | legal Q&A + medical transcripts | ✗ synthetic | 7 formal (GLiNER2-PII benchmark subset = 200 docs) | ✗ none | [SPY NAACL-SRW 2025](https://aclanthology.org/2025.naacl-srw.23/); used by [zaratiana2026_gliner2_pii](../../research-wiki/papers/zaratiana2026_gliner2_pii.md) ([arXiv 2605.09973](https://arxiv.org/abs/2605.09973)) |
| **PIIBench** (Jha, corpus) | 10 sources | mixed | 48 types / 97 BIO; geo/org via absorbed NER | partial (geo/org; ages/occupations UNVERIFIED) | corpus [arXiv 2604.15776](https://arxiv.org/abs/2604.15776); method [jha2026_piibench_deberta](../../research-wiki/papers/jha2026_piibench_deberta.md) ([arXiv 2605.25816](https://arxiv.org/abs/2605.25816)) |
| **Gretel** (finance / masking-en-v1) | finance / multi-domain | ✗ synthetic | ~20–51 (finance-skewed; en-v1 span availability UNVERIFIED) | Apache-2.0 | [HF gretelai](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) |
| **CoNLL-2002 / 2003** | newswire | ✔ real | PER/ORG/LOC/MISC | LOC only | [W02-2024](https://aclanthology.org/W02-2024.pdf) / [W03-0419](https://aclanthology.org/W03-0419.pdf) |
| **WikiNER** | Wikipedia | ✔ real (silver, noisy) | PER/ORG/LOC/MISC | LOC only | DOI [10.1016/j.artint.2012.03.006](https://doi.org/10.1016/j.artint.2012.03.006) |
| **MultiNERD** | Wikipedia/WikiNews | ✔ real (silver) | 15 types (PER/LOC/**TIME**/ORG + world-knowledge) | LOC + TIME | [2022.findings-naacl.60](https://aclanthology.org/2022.findings-naacl.60/); **used — our P7 generality probe** |
| **PII-Bench** (hyphenated) | query-aware QA | ✗ synthetic | 55 categories; QA-style, not spans | n/a | [arXiv 2502.18545](https://arxiv.org/abs/2502.18545) |

## Datasets used in this project (local)

`corpora/` holds two kinds — **PII/anonymization** (with identifier gold) and **utility-task** corpora
(round-trip evaluation, *no* PII gold). Only TAB carries direct/quasi labels.

| Local path | What | Gold | Used for |
|---|---|---|---|
| `corpora/tab/echr_{train,dev,test}.json` | TAB (1,014 / 127 / 127 ECHR) | DIRECT/QUASI, 8 types | detector train (Arm B) + gate; the privacy-ceiling corpus |
| `data/pii_span_dataset/{train,dev}.jsonl` | TAB → gliner windows (11,022 / 927) | derived from TAB | Arm-B fine-tuning input (`build_pii_span_dataset.py`) |
| MultiNERD-en (HF, not vendored) | general NER, out-of-TAB types | 15 NER types | **P7 zero-shot generality probe** (`pii_zeroshot_generality.py`) |
| `corpora/synthpai/train.jsonl` | SynthPAI ([arXiv 2406.07217](https://arxiv.org/abs/2406.07217)) | 8 person attributes | pipeline corpus + re-identification attacker; span routing (FIG 02) |
| `corpora/clinical/aci.jsonl` | ACI-Bench doctor-patient dialogues ([yim2023_acibench_visit_note_generation](../../research-wiki/papers/yim2023_acibench_visit_note_generation.md)) | task gold (note) | round-trip utility (clinical), FIG 03/04; qualitative detector demo |
| `corpora/clinical/mts.jsonl` | MTS-Dialog ([benabacha2023_mtsdialog_clinical_note](../../research-wiki/papers/benabacha2023_mtsdialog_clinical_note.md)) | task gold | round-trip utility (clinical) |
| `corpora/enron/replies.jsonl` | Enron email replies | task gold (reply) | round-trip utility (email), FIG 03/04; detector demo |
| `corpora/aeslc/test.jsonl` | AESLC email subject lines ([zhang2019_aeslc_subject_line_generation](../../research-wiki/papers/zhang2019_aeslc_subject_line_generation.md)) | task gold | utility-task corpus |
| `corpora/cnndm.jsonl` | CNN/DailyMail summarization | task gold | utility-task corpus |
| `corpora/wikibio/val.jsonl` | Wikipedia biographies ([arXiv 2205.06895](https://arxiv.org/abs/2205.06895)); 400–4000-char band of the vendored NR set | proxy (first sentence) | round-trip QA (bios); span-dense, **viable** (dev-log 2026-07-06) |
| `corpora/qmsum/val.jsonl` | QMSum committee excerpts (specific-query spans; [arXiv 2004.13822](https://arxiv.org/abs/2004.13822)) | task gold (query answer) | round-trip QA (meetings); **QA desert** under summary task — 86% ceiling-reject, dropped (dev-log 2026-07-06) |
| ai4privacy pii-masking-400k (referenced, not vendored) | synthetic formal PII | 17 types | planned auxiliary mix (formal-type breadth), not yet used |

## Gaps and recommendations

- **True QUASI supervision is scarce: only TAB + the Wikipedia-bio set.** Everything else is either
  mask-all (quasi-typed but no direct/quasi semantics), attribute-level, or formal-PII-only.
- **Best next training addition → Wikipedia biographies ([arXiv 2205.06895](https://arxiv.org/abs/2205.06895)).**
  Only true DIRECT/QUASI complement to TAB, real text, different domain (bios) — would cut the
  legal-domain overfit and is the natural candidate to test whether mixing diverse quasi text
  mitigates the Arm-B P7 generality erosion (`learned-PII-detection.md` §5.3).
- **Honest real-domain test beyond TAB** → the Wikipedia-bio test split (open); i2b2-2014 (real,
  quasi-typed) if the DUA is worth it. SynthPAI remains the re-identification-attacker eval.
- **Auxiliary formal-type breadth** (PERSON/CODE/LOC) → Nemotron-PII (quasi-rich) or ai4privacy — but
  synthetic, so for robustness, not transfer.
- **The synthetic-transfer caveat is field-wide.** Almost every open dataset is machine-generated;
  held-out F1 on them overstates real-world protection. TAB, Wikipedia-bio, and the (gated) clinical
  corpora are the only real-text, quasi-typed, span-annotated options.

## Sources

Survey conducted 2026-07-04 (two research passes); arXiv/DOI/HF identifiers verified at survey time,
UNVERIFIED items flagged inline. Registered wiki pages linked above carry their arXiv ids; unregistered
datasets are cited by arXiv/DOI/HF URL. Detector-side context and the QUASI-gap argument:
[`learned-PII-detection.md`](learned-PII-detection.md). Not yet registered as research-wiki pages
(candidates): Wikipedia-bio (2205.06895), SynthPAI (2406.07217), REDACT (2606.19881), i2b2-2014,
RAT-Bench (2602.12806).
