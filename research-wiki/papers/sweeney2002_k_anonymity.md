---
type: paper
node_id: paper:sweeney2002_k_anonymity
title: "k-Anonymity: A Model for Protecting Privacy"
authors: ["Latanya Sweeney"]
year: 2002
venue: "International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems 10(5)"
external_ids:
  arxiv: null
  doi: "10.1142/S0218488502001648"
  s2: null
tags: ["k-anonymity", "generalization-hierarchy", "quasi-identifier", "suppression",
       "structured-anonymization"]
added: 2026-07-04T00:00:00Z
---

# k-Anonymity: A Model for Protecting Privacy

## One-line thesis
A release is k-anonymous when every record is indistinguishable from at least k−1 others on
its quasi-identifiers; achieved by generalization over per-attribute hierarchies plus
suppression — privacy as a structural property of the released values, checked by counting,
with no model at release time.

## Problem / Gap
Removing direct identifiers does not anonymize: combinations of innocuous attributes
(quasi-identifiers) link released records to external sources. Sweeney's canonical linking
attack re-identified the Massachusetts governor's health record by joining "anonymized"
hospital data with the public voter roll on {ZIP, birth date, sex} (a companion study
estimated ~87% of the US population is unique on that triple).

## Method
- **Quasi-identifier**: the attribute set an attacker can plausibly link on.
- **Domain generalization hierarchy (DGH)**: per attribute, a lattice of increasingly coarse
  values (ZIP 02139 → 0213* → 021** → *; birth date → month → year → decade).
- **k-anonymity**: every released tuple's quasi-identifier value must occur ≥ k times in the
  release; enforced by walking values up the DGH (generalization) and deleting outliers
  (suppression). Includes MinGen, a minimal-distortion (NP-hard in general) generalization
  algorithm.

## Key Results
- Formal definition and proof that generalization+suppression over DGHs achieves the property
  with bounded distortion; the linking-attack demonstration motivating quasi-identifiers.
- The enforcement check is **model-free counting**: legality of a released value = size of its
  consistency set. Protection is auditable by inspection of the release alone.

## Assumptions
- The attacker links via equality on quasi-identifiers against an external table; a uniform
  prior over the k-block (re-identification probability ≤ 1/k assumes the attacker cannot
  rank the k candidates).
- Quasi-identifier set is known in advance; data is relational/structured.

## Limitations / Failure Modes
- Attribute disclosure survives identity protection: a homogeneous k-block leaks the sensitive
  value without re-identification (→ l-diversity, Machanavajjhala et al. 2007) and skewed
  distributions leak probabilistically (→ t-closeness, Li et al. 2007).
- Non-uniform priors break the 1/k bound (famous/likely candidates dominate the block).
- Syntactic, not compositional: two independently k-anonymous releases can jointly
  re-identify. Optimal generalization is NP-hard.

## Reusable Ingredients
- DGHs are exactly this project's per-type generalization lattices; k-anonymity's
  count-based legality is a zero-model inference-time mask: risk(level) ≈ 1/|candidates
  consistent with level|, thresholded per type.
- The uniform-prior caveat prescribes population-weighted counts or an attacker-shootout
  calibration margin rather than raw counts.

## Open Questions
For free text: what is the candidate universe per span type (gazetteer for LOC/ORG, ontology
for conditions, interval width for dates/quantities), and how well does 1/k correlate with a
real LLM attacker's hit rate on lattice levels?

## Relevance to This Project
Surfaced while designing the removal of `walk_risk` (a Pythia-410m contrastive probe) from
the HarnessCloak inference path (see `docs/research/inference-risk-enforcement.md`): the
structural-lattice-risk option is k-anonymity transplanted to span substitution — risk as a
counted property of the lattice node, enforced by lookup at inference, with the LLM attacker
retired to offline calibration. Bears directly on the substitution-architecture design
question (tau mask semantics) and on the project's user-specified-lattice positioning. Its
known failure modes (attribute disclosure, non-uniform priors) name exactly the residual
channels our evaluation attacker must price (the measured famous-context recovery of "LJM2"
is the non-uniform-prior failure in the wild).
