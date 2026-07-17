# QA-v2 audit validation brief

## Goal
Validate the new two-stage QA-v2 audit reports on the first available ACI documents,
`aci/D2N001` through `aci/D2N007`, after a fresh QA-v2 detector/arms and ranker-environment build.

## Constraints
- `aci/D2N000` is absent from the corpus; do not silently substitute another document.
- Build one GPU process at a time and write new artifacts under `/tmp`.
- Do not call the relation teacher or reader in this validation; this evaluates the pre-teacher
  environment report and its metadata handoff only.

## Success criteria
- The arms build writes deterministic environment audit JSON, JSONL, and Markdown sidecars.
- The report exposes unprofiled profile-backed spans, semantic profile matches, coarse menus,
  containment/profile conflicts, and detector rejections with usable evidence.
- Manual review can distinguish expected detector noise from plausible lattice/data defects without
  false claims that the report is a correctness oracle.

## Fixed design decisions
- Use the QA-v2 clinical detector preset and existing profile embedding index.
- Build exactly seven documents with `--n-docs 7 --corpora aci`.
- Build the downstream ranker environment with `--skip-probes`; QA teacher/reader validation follows
  separately if the environment audit is useful.
