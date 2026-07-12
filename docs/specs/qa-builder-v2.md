---
type: reference
status: partial
created: 2026-07-12
updated: 2026-07-12
tags: [qa, reward-design, probe-builder, span-linking, credit-assignment, codesign, draft]
companion: [docs/specs/RL/interactive-ranker-v2.md,
            docs/specs/RL/interactive-ranker-v2-decision-log.md,
            docs/specs/RL/training-task-env.md]
---

# QA builder v2 — ranker-reward codesign draft

**Status: draft for a later dedicated design pass.** This document records requirements and
codesign opportunities discovered while redesigning the interactive ranker. It does not yet
replace the current QA-builder specification or implementation.

## Problem

The current builder rejects many generated QA pairs, and accepted coverage is uneven across
documents and detected spans. Ranker training must not interpret missing probes as evidence
that a span has no task utility. At the same time, accepted probes are the cheapest available
structure for routing document-level utility toward the actions most likely to affect it.

QA-builder v2 and ranker v2 therefore need a shared interface: the builder supplies validated
measurement items and provisional dependency links; the ranker preserves global credit and
uses counterfactuals to correct those links rather than treating them as causal labels.

## Pinned codesign requirements

### Shared frozen span artifact

- Run the composed detector once and assign stable span IDs.
- QA construction and RL consume the same frozen rows, including offsets, type, surface,
  overlap resolution, detector provenance, and detector version.
- Never re-detect independently inside either pipeline.

### Explicit dependency links

- Give the QA teacher a numbered inventory of detected spans.
- Require `depends_on_span_ids` and supporting `depends_on_quotes` in generated probes.
- Validate that every ID exists and every quote supports the referenced span.
- Keep canonical substring inference only as a migration fallback for legacy artifacts.
- Represent multi-span questions as hyperedges rather than forcing one primary span.

### Missing coverage is explicit

- Report accepted probes per span, per type, per document, and per corpus.
- Distinguish no generated candidate, generation failure, ceiling rejection, floor rejection,
  lint rejection, reader instability, and unlinked dependency.
- Emit the set of ranker-controlled spans with no accepted linked probe.
- Do not silently remove those spans from RL or assign them zero utility.

### Probe outputs support structured credit

Each accepted probe should expose at least:

```text
probe_id
kind
weight_class
span_ids
question / options / gold
ceiling and floor validation evidence
per-rollout score
builder and reader pins
```

The ranker reward module must be able to partition linked probe components from unlinked and
document-global components without double counting.

## Codesign opportunities for the later overhaul

### Coverage-aware generation

Allocate generation effort toward spans and interaction sets lacking accepted probes rather
than producing a fixed number per document. Hard spans may receive retries or alternative
question forms, but retry budgets and acceptance rules must be pinned before evaluation.

### Counterfactual validation during construction

Ceiling/all-placeholder anchors establish broad support but do not prove a probe depends on
every declared span. On a bounded subset, replace one linked span at a time and retain the
observed dependency signature. This can estimate link precision and identify probes whose
answer changes only through a different span.

### Interaction probes

Permit probes whose dependency set contains multiple spans when the downstream decision
genuinely requires their relationship. Record the full set and avoid manufacturing singleton
credit. Ranker-side contextual counterfactuals remain responsible for marginal attribution.

### Adaptive counterfactual scheduling

Expose builder uncertainty and coverage metadata so RL can prioritize expensive
counterfactuals for:

- spans without accepted probes;
- probes with uncertain or multi-span links;
- high-entropy policy decisions;
- action menus whose utility slope is unresolved;
- detector types with known precision problems.

## Non-goals for this draft

- Choosing a new teacher or reader model.
- Changing probe weights or acceptance thresholds.
- Claiming that `span_ids` are causal annotations.
- Solving detector precision and recall inside the QA builder.
- Making QA coverage a prerequisite for a ranker-controlled span to participate in RL.

## Questions reserved for the dedicated design pass

- What minimum coverage target is useful without turning it into an invalid quality claim?
- Which rejection classes should trigger regeneration rather than permanent exclusion?
- How much single-span counterfactual validation is affordable during probe construction?
- Should probe generation target lattice levels, downstream decisions, or both under one
  shared budget?
- How should reader disagreement and instability be represented in training weights?

