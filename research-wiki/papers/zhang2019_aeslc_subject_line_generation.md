---
type: paper
node_id: paper:zhang2019_aeslc_subject_line_generation
title: "This Email Could Save Your Life: Introducing the Task of Email Subject Line Generation (AESLC)"
authors: ["Rui Zhang", "Joel Tetreault"]
year: 2019
venue: "ACL 2019"
external_ids:
  arxiv: "1906.03497"
  doi: null
  acl: "P19-1043"
tags: ["benchmark", "email-generation", "task-utility", "pii-rich", "enron", "eval-corpus"]
added: 2026-07-02T00:00:00Z
---

# This Email Could Save Your Life: Introducing the Task of Email Subject Line Generation (AESLC)

## One-line thesis
Introduces email subject-line generation — abstractively summarizing an email body into a short subject —
built from the Enron corpus with crowdsourced multi-reference annotations.

## Method / Contents
Annotated Enron Subject Line Corpus (AESLC): email body → subject pairs mined from the Enron corpus (517k
messages, 150 mailboxes), filtered to bodies with ≥3 sentences / ≥25 words, first email per thread only. Dev
and test subjects carry 3 human-annotated references each. Proposes a two-stage extract-then-rewrite approach
with a subject-quality-estimation reward.

## Key Results
- First dataset + task for email subject generation; multi-reference eval (ROUGE against 3 references).
- The Enron base makes it a standing source of **real** PII (real people, orgs, projects, dates, amounts),
  widely reused for PII-extraction and privacy studies.

## Limitations / Failure Modes
- Subject lines are very short, so they surface only a subset of the body's entities — a lighter "restatement"
  load than a full email reply/body-generation task on the same corpus (which is the fuller-PII variant).

## Relevance to This Project
**Why surfaced:** the non-medical (email) domain in the two-domain task-oriented eval adopted for the
LatticeCloak next stage (§5.4, [`docs/html/LatticeCloak.html`](../../docs/html/LatticeCloak.html)). **Fit:**
Enron is real-PII text; a generated subject/reply must reproduce names, orgs, dates and amounts — exactly the
direct-placeholder and quasi spans LatticeCloak substitutes — so inversion is exercised and coarsening carries
measurable utility cost, with multi-reference gold for scoring. Second domain alongside clinical notes; the
Enron base also anchors the attacker axis (Enron appears in LLM-PBE). Spec:
[`docs/specs/benchmarks.md`](../../docs/specs/benchmarks.md).

## Connections
_Edges recorded in `graph/edges.jsonl`._ Non-medical complement to
[`yim2023_acibench_visit_note_generation`](yim2023_acibench_visit_note_generation.md) and
[`benabacha2023_mtsdialog_clinical_note`](benabacha2023_mtsdialog_clinical_note.md).
