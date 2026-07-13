---
type: dev-log
status: current
created: 2026-07-13
updated: 2026-07-13
tags: [rl-v2, qa-builder, relation-teacher, debugging]
---

# QA teacher v2 debugging notes

## Context candidate inventory: intended role

The relation compiler must distinguish between two kinds of arguments:

- **Linked controlled arguments** are detected occurrences backed by a policy decision. They need a legal lattice action and participate in the joint representative anchor.
- **Source-grounded context arguments** are exact spans such as a lab, a referral destination, status, or category. They are validated against the source but do not receive a policy decision or lattice action.

The context candidate inventory was introduced so the teacher could name the second kind without creating a fake detected occurrence. Its intended example was a controlled condition linked to an uncontrolled test literal.

## Why the current inventory is not useful to the teacher

The present implementation is a narrow cue-regex extractor. On D2N002 it produces two separate entries for `an autoimmune panel`, plus `thyroid labs`, `a thyroid panel`, and `physical therapy`. It does not express which nearby controlled span each literal might relate to, duplicates source content already visible in the document, and is repeated again in the evidence-window representation.

The teacher receives opaque hash identifiers, character offsets, and a global list detached from its local source context. This creates clerical work instead of improving discovery. The list is therefore currently a distraction, even though source-grounded uncontrolled arguments remain necessary in the compiler.

## Replacement direction

Do not expose a global context-candidate inventory to the teacher. Keep the internal exact-span inventory for deterministic validation, but present a compact local context label only inside a relation-focused evidence card, for example `C3: thyroid labs (test)`. The card should also contain nearby controlled span labels and the source excerpt. Code maps those short labels back to stored occurrence/context records and offsets.

## Revisit trigger

Reintroduce a separately visible context inventory only if evidence-card-local labels cannot express a valid source-grounded argument, such as a relation whose two arguments must be linked across non-adjacent authoritative source regions. Before doing so, demonstrate on a real smoke that the added inventory raises accepted, source-grounded relation coverage without increasing invalid links, duplicate attempts, protected-term leakage, or teacher abstention.
