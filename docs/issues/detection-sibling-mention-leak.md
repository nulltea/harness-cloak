---
type: research
status: current
created: 2026-07-04
updated: 2026-07-04
tags: [issue, detection, coref, recall, leak-channel, attacker, sibling-mentions]
companion: [../specs/RL/surrogate-ranker-infiller.md, ../research/learned-PII-detection.md,
            ../specs/attacks.md]
---

# Issue: retained sibling mentions are the dominant attacker-recovery channel

**Measured (2026-07-03 probe shootout, frontier referee gemini-3.1-pro: hit@5 ≈ 0.49 on engaged
fills):** inspecting the recovered examples, most successes exploit **context the substitutor
left in cleartext**, not weak generalization levels:

- "gastroenterology" substituted ("a life science") while "**the gastroenterologist**" stands
  cleartext one sentence earlier — attacker reads the answer off the sibling mention.
- "mount baker area" → "a location", while "**northern cascades**" survives one clause earlier.
- "washington" generalized while "northern cascades / mount baker" remain — the state is pinned.

**Root cause (two parts):** (1) **detection recall** — GLiNER's zero-shot labels fire on some
surface forms of a concept and not others (department "gastroenterology" detected as ORG-ish;
profession noun "the gastroenterologist" missed — the measured DEM 0.56 recall gap);
(2) **no concept-level aliasing** — `coref_chains` links repeated PERSON name mentions only;
nothing ties "gastroenterologist" ↔ "gastroenterology" or "mount baker area" ↔ "northern
cascades" into one substitution decision.

**Impact:** this channel is **upstream of level selection** — no τ-walk fix or RL-trained ranker
can close it, because the leaking span was never in the action space. It caps realized privacy
for the whole substitutor workstream and must be reported as such (detection recall = the
privacy ceiling, per project rules).

**Disposition (decided 2026-07-04):** recorded as the measured dominant leak channel; **not** on
the RL critical path. Owned by the detector workstream
([learned-PII-detection.md](../research/learned-PII-detection.md) — supervised GLiNER finetuning
for the recall gap; concept-level alias/coref is a design item there). The RL evaluation should
report attacker successes split by channel (retained-context vs fill-inference) so the ceiling
is attributed correctly.
