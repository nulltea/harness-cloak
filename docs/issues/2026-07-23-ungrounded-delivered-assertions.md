---
type: research
status: current
created: 2026-07-23
updated: 2026-07-23
tags: [issue, utility-assertions, qa-builder, reward-ceiling, data-quality]
companion: [../specs/RL/ranker-v2-architecture.md]
---

# Issue: delivered assertions not grounded in the dialogue (found via aci/D2N048)

## Addendum 2026-07-23: two sharper classifications (BC-smoke follow-up)

1. **Policy-invariant delivered assertions.** The environment controls no AGE/SEX
   occurrences anywhere (policy types: drug/health-condition/medical-procedure/LOC;
   fixed types: PERSON/CODE placeholders). DEMOGRAPHIC `field_value` assertions are
   therefore invariant to every substitution action — they measure only the remote
   model's note-writing, contribute a constant to the policy reward (cancels in
   leave-one-out advantage, pollutes ceilings/aggregates/gate thresholds). Refinement:
   classify delivered assertions by whether their evidence overlaps a controlled
   occurrence; keep policy-invariant ones in the document-utility report but exclude
   them from the ranker's reward denominator.
2. **Detector coverage gap (privacy leak).** `aci/D2N048` is one of 4/67 documents with
   zero fixed decisions: the span detector produced no PERSON span for the lowercase
   dialogue name ("brittany"), so the patient first name reaches the remote model
   verbatim. The other 63 docs carry 124 PERSON + 84 CODE placeholder fixed decisions,
   so this is a detection miss on lowercase transcript names, not a scope decision.

## Symptom

BC smoke (2026-07-23, medgemma remote): `aci/D2N048` scores utility 0.1 for every action
vector. Of its 4 delivered assertions, 3 demand content that `doc_p` (the dialogue
transcript) cannot support:

- `field_value DEMOGRAPHIC age="76-year-old"` — "76" appears nowhere in the transcript;
- `field_value DEMOGRAPHIC sex="female"` — "female" appears nowhere in the transcript;
- `contains "bruising"` — the only occurrence is an off-topic aside ("learned the very
  bruisy way that racquet ball wasn't for me"), not a clinical finding.

These presumably came from the ACI reference note's demographics header / exam section,
which the dialogue never states. Any remote model fails them; the document's reward
ceiling is ~0.25 (weighted 0.1) by construction, independent of the policy.

## Impact

Documents with ungrounded delivered assertions contribute constant near-zero utility —
pure noise in BC document points, ExIt winner selection (all candidates tie), and
hybrid advantage estimates. Unknown how many of the 67 docs are affected; the
calibration preflight's ceiling anchors will surface the full list.

## Proper fix

QA-builder should gate delivered assertions on transcript groundedness (the same
three-point-gate idea used for context assertions): a delivered fact must be recoverable
from the dialogue, not only from the reference note.

## Provisional handling

None needed for smoke/preflight — affected documents just carry a flat low ceiling.
Before interpreting full-run utility aggregates, exclude or reweight documents whose
ceiling anchor is itself near zero.
