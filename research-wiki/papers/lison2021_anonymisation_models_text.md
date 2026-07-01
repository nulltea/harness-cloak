---
type: paper
node_id: paper:lison2021_anonymisation_models_text
title: "Anonymisation Models for Text Data: State of the Art, Challenges and Future Directions"
authors: ["Pierre Lison", "Ildikó Pilán", "David Sánchez", "Montserrat Batet", "Lilja Øvrelid"]
year: 2021
venue: "ACL 2021"
external_ids:
  arxiv: null
  doi: "10.18653/v1/2021.acl-long.323"
  s2: null
tags: ["sok", "text-anonymisation", "pii", "disclosure-risk", "quasi-identifiers", "rd0", "rd2"]
added: 2026-07-01T00:00:00Z
---

# Anonymisation Models for Text Data: State of the Art, Challenges and Future Directions

## One-line thesis
Position/SoK paper bridging NLP sequence-labelling anonymisation and privacy-preserving data
publishing; argues for moving beyond span tagging toward explicit disclosure-risk modelling.

## Key Results
- Unifies two disconnected fields (NLP anonymisation vs. privacy-preserving data publishing).
- Names three open challenges: (1) accounting for **multiple types of semantic inference**,
  (2) the **disclosure-risk vs. utility** balance, (3) how to **evaluate** anonymisation quality.
- Case for incorporating an explicit disclosure-risk measure into the anonymisation process rather
  than relying on sequence labelling alone.

## Relevance to This Project
**Why surfaced:** SoK anchor for **RD0 (protectable units)** and **RD2 (PII detection & sensitivity
typing)** in [`docs/research/beyond-rantext.md`](../../docs/research/beyond-rantext.md). **Relevance:**
its central thesis — span tagging is insufficient, model the *combination*/inference risk — is exactly
the A1/F4 correlation problem from [`rantext-limitations.md`](../../docs/research/rantext-limitations.md):
RANTEXT's per-token, surface-level treatment is precisely the "sequence labelling without disclosure-
risk" pattern this SoK argues against.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Abstract (original)

> This position paper investigates the problem of automated text anonymisation, which is a
> prerequisite for secure sharing of documents containing sensitive information about individuals. We
> summarise the key concepts behind text anonymisation and provide a review of current approaches.
> Anonymisation methods have so far been developed in two fields with little mutual interaction,
> namely natural language processing and privacy-preserving data publishing. Based on a case study, we
> outline the benefits and limitations of these approaches and discuss a number of open challenges,
> such as (1) how to account for multiple types of semantic inferences, (2) how to strike a balance
> between disclosure risk and data utility and (3) how to evaluate the quality of the resulting
> anonymisation. We lay out a case for moving beyond sequence labelling models and incorporate
> explicit measures of disclosure risk into the text anonymisation process.
