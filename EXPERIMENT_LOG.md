# QA-v2 audit validation log

## Current hypothesis
The environment audit is structurally sound but needs triage tiers: it should preserve full raw
evidence while prioritizing the small set of likely lattice/matcher defects over expected abstentions
and policy-exhausted rows.

## Latest result
`aci/D2N001`–`aci/D2N007` build into a seven-document arms artifact and a 75-decision ranker
environment (with `--skip-probes`). The environment sidecars coexist safely. The audit join now maps
post-admission clinical `raw_label` values through the detector's label registry, reducing false
detector-to-walk-drop reports from 26 to 2. The resulting report has 191 events: 79 unprofiled
profile-backed spans, 27 low-confidence condition rejections, 75 policy-exhausted profiled spans,
five semantic matches, two containment/profile conflicts, one coarse semantic menu, and two genuine
walk drops. High-signal review candidates are `abdominal distension` -> `abdomen` with only
`gastrointestinal condition`, and the split `type 2 diabetes`/`diabetes` profiles.
An uncontended rerun produced the same event categories and counts. Its byte-level audit hash changed
only because detector/NLI scores drifted at small floating-point precision; the corresponding event
identities and decisions remained the same. Treat the hash as an artifact-content fingerprint, not a
cross-run deterministic identifier until score quantization is designed deliberately.
The approved environment diagnostics now add one canonical-profile ladder record per controlled
profile-backed decision, a cross-profile coreference candidate monitor, and one logging-only
self-type NLI score per controlled profile decision. A real D2N001 build emitted 11 ladder records,
11 self-type scores, and one expected class-instance candidate (lisinopril -> blood pressure
medication), without changing detector admission or actions. The current embedding index is stale
after profile edits, so semantic matching correctly degraded to exact-only for this validation.

## Next step
Regenerate the embedding index after the approved profile edits, then rebuild the seven-document
environment artifact and inspect the three new diagnostics together. Run the teacher/reader QA build
only after approving the external-call cost and after the environment report is considered useful
enough to guide it.
