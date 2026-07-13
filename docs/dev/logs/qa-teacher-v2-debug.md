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

## 2026-07-13/14 — r16 root cause and the v6 contract (live-debugged on D2N002)

The r16 smoke (14 eligible spans, 12,269-char prompt) proposed one relation, expressed both
arguments as `kind: "context"` literals, and compiled to zero. The OpenRouter UI reasoning trace
gave the cross-boundary root cause:

1. **Reasoning cap truncated the source scan.** The 1,024-token reasoning budget cut off at
   source offset ~5495 ("i also wan na go…"), before the region at 5899+ holding the Synthroid
   continuation, thyroid-labs order, and physical-therapy referral. Hence one relation and an
   all-exhausted ledger. Completion caps had separately produced empty replies. The v6 contract
   sends **no token caps** (only the reasoning-trace exclusion).
2. **The prompt never named the record fields.** The trace plans a text format ("The format? Not
   fully specified… we need to infer") such as `prescribed_with(S3, S9)`; the constrained decoder
   then demanded fields the plan never chose, and the model fell into the all-null context
   branch. The v6 RESPONSE section names `span_label`/`support_property`/`literal`/kind rules and
   includes record-shaped examples.
3. **The wire schema permitted an always-rejected shape.** A zero-linked argument pair was
   wire-legal but compiler-fatal (`missing_linked_argument`). The v6 schema binds the pair to
   `linked+linked` / `linked+context` / `context+linked` with roles fixed by position.

Three further live v6 draws exposed and fixed a second layer, each with a regression test:
per-duplicate-label fact inflation (new `duplicate_mention` ledger state + emit-at-the-relation-
sentence rule), empty ledger reasons (wire `minLength`), protected surfaces in ledger reasons
(compiler sanitizer), drug literals untypeable by the closed lexical rules (relation-slot class
typing, with `protected_context_literal` rejection when a literal lands on a controlled span),
"ca n't take … because of" contraindications and "exacerbation of" attributions missing from cue
sets, cue windows that ignored cues preceding the subject, spoken-ellipsis clause splitting, and
teacher QA that names surfaces instead of levels (deterministic surface→selected-level
substitution before the leakage gates, recorded as `sanitized_qa`; level tokens and
placeholder-label tokens exempt from overlap lint).

Replaying the best live draw through the final compiler yields 4 accepted relations across 3
types with sanitized level-based QA; the honest rejections that remain are a cross-turn referral
(spec-rejected by design) and a `thyroid labs` answer whose token overlaps the chosen level.

## 2026-07-14 — D2N002 relation-coverage ceiling audit (span-by-span)

Why a good v6 draw tops out at ~6 relations despite many generalizable spans: 21 of 35 frozen
occurrences are `controlled: false` (no lattice profile → no ranker decision), so prefilter rule
2 excludes them from `S#` labels; they are reachable only as quoted literals beside a linked
argument in the same/adjacent clause. Audit of every dropped span against the source:

**Real relations currently unreachable:**

| Span | Statement | Relation | Blocking rule |
|---|---|---|---|
| `immunosuppressive medications` @2008 | "you're taking your immunosuppressive medications?" | prescribed_with(kidney transplant → immunosupp.) | Nearest transplant mention ~4 turns away with non-backchannel turns between: a discourse/topic link the cross-turn prohibition rejects by design. |
| `polycystic kidneys` @1720 | "you've had the kidney transplant a few years ago for some polycystic kidneys" | transplant-for-condition (indication) | Same clause and explicit, but transplant is detector-typed `health-condition` (blocks `treated_with`'s procedure slot) and the "<procedure> for <condition>" indication form matches no closed cue/connector. |
| `tylenol` @1036 | "what have you taken for the pain? a little tylenol" | drug-for-symptom | "knee pain"/"the pain" is not a detected span; no linked argument exists, and one is required. Detector coverage. |
| `immunocompromised` + WBC @4951 | "white blood cell count is not elevated … concerned … in somebody who's immunocompromised" | monitoring rationale | Both ends uncontrolled; WBC detected as `CODE` junk (tracked detector-misclassification issue). |

**Verified correct abstentions:** `edema`/`erythema` (exam findings, no ontology relation),
`heart exam`/`physical examination` (murmur undetected, finding not a relation), `imaging` and
the first `physical therapy` mention ("we'll talk about … possibly referral" — hypothetical),
demographics/DATETIME, `acute exacerbation` (fact already carried by arthritis → knee pain),
`synthroid` (uncontrolled but reachable as a literal; good draws captured it).

Conclusion: the expressible ceiling (~6–7) is set by (a) detector/lattice coverage of the
frozen environment, (b) the ≥1-linked + same/adjacent-clause grounding contract, and (c) closed
cue sets missing the indication form — not by the relation teacher.

## 2026-07-14 — coverage extension after the audit (v7/r19, compiler v5)

From the four unreachable relations, one was soundly extendable now: the polycystic-kidneys
indication. Implemented as a closed procedure-form lexicon (a condition surface naming a
performed procedure also fills procedure slots; prompt shows `condition/procedure`) plus a
reversed `treated_with` connector for "<procedure> for <condition>". Pins bumped to prompt v7 /
revision r19 / assertion-compiler v5. Deliberately **not** extended: backchannel-turn adjacency
for the physical-therapy referral (the reference states it conditionally — accepting would
overstate the truth source); the immunosuppressive link (needs the reference-grounded channel,
logged as an open fork in the decision log); tylenol/WBC (detector coverage, tracked in the
misclassification issue).

**Open finding — provider draw variance.** `nvidia/nemotron-3-super-120b-a12b:free` is
nondeterministic at `temperature=0.0` even with `seed`: identical prompts returned 6, 4, 1, 3,
and 4 proposals of varying quality across draws. The production builder caches the first draw
per pin, so build quality is currently a lottery. Options (user decision, none implemented):
accept single-draw variance as an environment property; permit one bounded revalidation call
only on an invalid reply (`ledger_inconsistent`/parse failure) as a validity rule, distinct from
coverage-chasing retries; or pin a non-free deterministic route (requires approval).
