---
type: plan
status: current
created: 2026-08-04
updated: 2026-08-04
tags: [rl, ranker-v2, lexicographic, exact-ties, multi-objective, gate, utility, privacy]
companion: [docs/specs/RL/ties-by-design.md,
            docs/specs/RL/interactive-ranker-v2.md,
            docs/specs/RL/interactive-ranker-v2-decision-log.md,
            docs/research/tie-ownership-root-cause-and-solution-space.md]
---

# Epsilon-Zero Lexicographic Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a document-level `epsilon = 0` lexicographic selector gate that decides whether exact utility ties contain reproducible count-based privacy gains and whether additive tie control should be retired.

**Architecture:** First construct the same deterministic five-anchor candidate slate for every retained document, then join that slate to fully pinned, complete action-vector results from the existing utility cache. A corpus-level verdict is permitted only when every primary document has complete standardized slate coverage and at least two count-distinguishable slate vectors. For each support-complete document, construct the exact measured-utility argmax set, select the highest profile-count score inside that set, and compare it with a privacy-blind utility-only tie-break and archived additive-controller behavior. The gate operates on complete document vectors, never structural dependency labels, per-decision approximations, observable-state aggregation, utility-tower logits, `alpha`, gain heads, or the invalid `0.044` threshold.

**Tech Stack:** Python 3.12, PyTorch only for archived policy replay, `decimal.Decimal` for exact cached-score equality, existing `UtilityCache`, `ProfileCountTargets`, ranker environment APIs, pytest, JSON/Markdown artifacts.

## Global Constraints

- The lexicographic slack is exactly `epsilon = 0`. No empirical floor, tolerance band, or user utility budget may enter admissibility.
- Floating-point representation tolerance is not semantic slack. Exact-set membership must use a canonical decimal weighted-utility key reconstructed from pinned component scores and weights.
- Selection is over complete legal document action vectors. Do not compose independently selected per-decision ties; span decisions interact through context, generation, injectivity, and output omission.
- Primary utility is the pinned QA-v2 document utility. Secondary privacy is the frozen profile-relative count score, used only as the experimental shaping/selection signal already permitted by the project.
- Structural dependency declarations and `policy_dependency_decision_ids` are excluded from tie membership. Their measured end-to-end precision is inadequate.
- The current additive controller, semantic privacy head, `alpha`, gain head, evidence hinge, cycle projection, softcap, KL anchor, and sensitivity regularizer are comparators only. None participates in the lexicographic selection.
- No reward calls are permitted in the cache-only gate. Missing vectors are reported, never generated silently.
- Candidate support is established by the deterministic five-anchor slate from `build_anchor_trajectories`, not by whichever vectors prior policies happened to cache. Missing standardized anchors are missing evidence, never zero opportunity.
- No method-level privacy claim may be made from count scores. A successful gate chooses the optimization/composition architecture; attacker-measured realized privacy remains a separate promotion requirement.
- The four repeated campaign documents are `aci/D2N005`, `aci/D2N027`, `aci/D2N031`, and `aci/D2N063`. The primary adjudication population is every other retained document; the four campaign documents are reported separately.
- Document is the statistical unit. Never micro-average cached contexts, action pairs, or vectors as independent samples.
- Keep the working tree's unrelated changes untouched. Do not stage or modify files outside this plan's declared file list.
- No new production dependency is allowed.

---

## 1. Decision This Gate Must Make

The gate is not another attempt to predict ties, calibrate a controller, or estimate a noise floor. It implements the target solution concept directly over already measured complete trajectories.

For document `d`, let `C_d` be the validated cached vectors from the frozen standardized five-anchor slate, `U_d(v)` their pinned document utilities, and `P_d(v)` their frozen profile-count scores. Other adaptively sampled cached vectors are an expanded-cache diagnostic only and never enter the primary gate.

\[
U_d^* = \max_{v \in C_d} U_d(v)
\]

\[
F_d^0 = \{v \in C_d : U_d(v) = U_d^*\}
\]

\[
v_d^{\mathrm{lex0}} = \arg\max_{v \in F_d^0} P_d(v)
\]

The equality in `F_d^0` is exact equality under the pinned cached utility representation. `0.044` does not appear anywhere.

The privacy-blind utility comparator is:

\[
v_d^{U} = \arg\min_{v \in F_d^0}
\left(\mathrm{Hamming}(v, v_d^{BC}),\; \mathrm{stable\_vector\_key}(v)\right)
\]

It maximizes utility first, then chooses the vector closest to the lambda-independent BC teacher without consulting count scores. This gives a deterministic primary-only baseline consistent with the existing lambda-zero design.

The free count opportunity is:

\[
G_d = P_d(v_d^{\mathrm{lex0}}) - P_d(v_d^U) \ge 0
\]

The gate answers four questions:

1. Is the primary-document candidate support broad and standardized enough to estimate `G_d` without zero-imputing cache misses?
2. Is the positive document-macro gain reproducible without any measured utility loss?
3. Does the selector eliminate dependence on arbitrary utility-logit scale and controller authority?
4. Does the observed opportunity justify replacing additive tie control with a set-level operator?

### What a pass means

A pass means: **the exact-tie portion of the product preference is lexicographic, materially present outside the campaign documents, and should no longer be implemented by additive logit control.** It authorizes the next architecture step: derive deployment behavior from a utility estimator/search procedure plus an exact feasible-set filter.

### What a pass does not mean

A pass does not establish that:

- the current tower can identify `F_d^0` on unseen documents;
- profile counts improve attacker-measured realized privacy;
- more than two lambda behaviors are available at `epsilon = 0`;
- a nonzero utility budget is justified;
- the observed cached candidate pool contains the global optimum over every legal vector.

It also does not treat a document with one cached vector, a missing standardized anchor, or no count-distinguishable standardized alternatives as evidence that `G_d = 0`.

At `epsilon = 0`, lambda semantics intentionally collapse to two regimes: lambda zero keeps the primary-only behavior; every positive privacy setting uses the same count-maximizing exact-utility set. Intermediate privacy settings require separately specified nonzero document budgets and are out of scope.

---

## 2. Literature Basis and Design Consequences

The plan follows the solution concept, not merely an algorithmic analogy.

1. **Lexicographic MORL:** [Skalse et al. 2022](../../research-wiki/papers/skalse2022_lexicographic_morl.md) ([arXiv 2212.13769](https://arxiv.org/abs/2212.13769)) formalizes maximizing the primary objective and optimizing lower-priority objectives only inside its optimal/slack-optimal set. Exact ties consume no primary-objective slack. Consequence here: exact measured utility defines admissibility; count never compensates for lower utility.
2. **Scalarization is not equivalent:** [Wray et al. 2015](../../research-wiki/papers/wray2015_lexicographic_mdp_slack.md) ([DOI 10.1609/aaai.v29i1.9647](https://doi.org/10.1609/aaai.v29i1.9647)) shows that lexicographic policies can lie outside the policy set obtainable by linear scalarization. Consequence here: failure to find a working `alpha` is not evidence that the weight needs further tuning.
3. **The utility model precedes the optimizer:** [Hayes et al. 2022](../../research-wiki/papers/hayes2022_practical_guide_morl.md) ([arXiv 2103.09568](https://arxiv.org/abs/2103.09568)) treats scalarization as a substantive model of user preference. Consequence here: “preserve measured task utility, then maximize privacy” must be represented as an ordered objective, not approximated because additive RL machinery is convenient.
4. **Global thresholds are unsafe:** [Tercan and Prabhu 2024](../../research-wiki/papers/tercan2024_thresholded_lexicographic.md) ([arXiv 2408.13493](https://arxiv.org/abs/2408.13493)) documents problems with global thresholded lexicographic value methods. Consequence here: the first gate uses no threshold at all. Any future nonzero slack must be document-budgeted and independently specified.
5. **Tie-breaking must be deterministic:** [Vamplew et al. 2024](../../research-wiki/papers/vamplew2024_value_function_interference.md) ([arXiv 2402.06266](https://arxiv.org/abs/2402.06266)) shows that random greedy tie-breaking aggravates value-function interference. Consequence here: both the utility-only and lexicographic selectors have complete deterministic tertiary ordering.

These papers support the operator family. They do not prove that the current count score corresponds to realized privacy or that a learned deployment utility estimator will recover the measured exact set. The gate keeps those claims separate.

### 2.1 Observable-state aggregation result

The cache aggregation follow-up does not alter the operator or supply gate labels. It found that observable-state aliasing is real, but the implemented statistic is a cache-uniform average over common cached suffixes, not policy-relevant `Q(s,a)` because it does not integrate action-conditional future suffix distributions. The result is also support-skewed: 3,929 of 4,080 aggregated groups come from the four campaign documents, while 3,329 of 4,122 observable-state groups are singletons.

Consequences for this plan:

- observable-state aggregation remains diagnostic only and cannot define utility admissibility, candidate support, or the gate verdict;
- the gate stays at complete document-vector level;
- sparse documents are classified as missing candidate support rather than assigned zero gain;
- a standardized policy-independent candidate slate is required before any corpus-level claim.

---

## 3. Adjudication Protocol

### 3.1 Candidate population

Use:

- environment: `results/ranker_v2/environment/ranker-env.json`;
- utility artifact: `results/ranker_v2/qa/aci-full.utility`;
- profile-count targets: `results/ranker_v2/reward/profile-count-targets.json`;
- utility cache: `results/ranker_v2/cache/utility-results.jsonl`;
- archived additive checkpoints supplied explicitly on the command line.

Before reading cache outcomes, build the expected standardized slate for every retained document with the existing `build_anchor_trajectories(document, count_reward)` contract. It deterministically constructs and exact-deduplicates these legal complete walks:

1. behavior-cloning teacher;
2. all-KEEP;
3. minimum-count non-KEEP;
4. midpoint authored level;
5. all-placeholder.

The slate may use frozen count targets to choose the minimum-count anchor, but it must not inspect utility outcomes, `G_d`, tower logits, controller behavior, or prior cache frequency. Archived lambda trajectories are comparators only and never satisfy slate support.

For each cache entry:

1. Require `reader_refresh == false`.
2. Require exact environment and utility-artifact hash matches.
3. Require every policy decision exactly once in environment order.
4. Replay legality through sequential claimed-fill/injectivity rules without invoking the learned policy.
5. Recompute float document utility from `component_scores`; require equality with stored `utility` within the cache contract's `1e-12` serialization tolerance.
6. Recompute the exact weighted-utility numerator with `Decimal(str(value))`; use this key for exact-set membership.
7. Recompute profile-count score from the frozen target artifact.
8. Deduplicate by ordered action vector. Conflicting base results for one vector are a hard validity failure.

Join every expected slate vector to the validated cache by `(doc_id, ordered_action_vector)`. `C_d` is exactly the joined standardized slate. Additional valid cached vectors are retained under `expanded_cache_diagnostic`, but cannot enlarge `C_d`, repair a missing standardized anchor, change `G_d`, or affect the verdict.

Assign exactly one `candidate_support_status` per document before computing utility gains:

- `unsupported-missing-slate`: at least one expected standardized vector is absent from the validated cache;
- `unsupported-no-count-contrast`: the complete standardized slate contains fewer than two unique vectors with different frozen count scores;
- `support-complete`: every standardized vector is cached and at least two have different frozen count scores.

After utility evaluation, assign `opportunity_status` only to support-complete documents:

- `supported-no-opportunity`: the measured exact utility argmax set has zero count spread or the privacy-blind comparator already chooses its maximum-count member;
- `supported-opportunity`: `G_d > 0`.

Every retained document remains in the coverage report. Only support-complete documents receive `G_d` and enter gain estimates. Unsupported documents receive `G_d: null`, never `0`, and are excluded from the bootstrap. A corpus-level architectural verdict requires every primary document to be support-complete; otherwise the cache-only run ends as `INSUFFICIENT-CANDIDATE-BREADTH` regardless of subset gains.

### 3.2 Primary and secondary populations

- **Primary:** all valid retained documents except the four campaign documents.
- **Campaign diagnostic:** the four campaign documents, reported separately.
- **All-document diagnostic:** primary plus campaign.

Do not tune any rule on the campaign documents and then report it as held-out evidence. The selector has no fitted parameters; the split exists to show whether the opportunity survives outside the documents that drove the controller campaign.

### 3.3 Required comparators

1. `utility_only_bc_nearest`: the privacy-blind exact-utility selector defined above.
2. `epsilon_zero_lexicographic`: the proposed selector.
3. `current_additive_greedy`: greedy lambda-zero and lambda-max vectors from each explicitly supplied archived checkpoint, joined to cache results when available.
4. `exact_set_privacy_min`: the minimum count score inside `F_d^0`, reported only as the lower edge of the available exact-tie interval, never as the primary comparator.

The additive comparator is report-only because cache misses can differ by checkpoint. The lexicographic decision must not depend on which archived controller happened to explore a vector.

### 3.4 Primary readouts

For every document report:

- candidate-vector count;
- expanded-cache diagnostic vector count and gains, explicitly marked non-adjudicating;
- expected standardized-slate size, cached standardized-slate size, missing anchor sources, `candidate_support_status`, and `opportunity_status`;
- exact-optimal-set size;
- exact-optimal count-score minimum, baseline, maximum, and spread;
- `G_d`;
- whether the selector changes the baseline vector;
- exact utility key and float utility of both selections;
- Hamming distance between selections;
- count provenance by action for the selected vector;
- additive comparator utility loss to `U_d^*` and count gap to `v_d^{lex0}` when cached;
- stable hashes of every selected vector and source artifact.

Aggregate with documents as units:

- primary support-completeness fraction and unsupported document IDs by reason;
- fraction of documents with `G_d > 0`;
- mean and median `G_d` over support-complete documents, including supported zero-opportunity documents but never zero-imputing unsupported documents;
- one-sided 95% document-bootstrap lower bound for mean `G_d` using 10,000 resamples and seed `20260804`;
- fraction of additive comparator vectors below `U_d^*`;
- fraction of utility-feasible additive vectors that fail to choose maximum count inside `F_d^0`;
- exact-optimal-set size distribution;
- standardized-slate coverage distribution by document, with adaptive expanded-cache coverage reported separately.

The fixed bootstrap seed controls reproducibility, not model behavior. The 95% level is the existing paired promotion confidence level and is not a utility-loss threshold.

### 3.5 Decision rule

Evaluate rules in order:

1. **INVALID:** any selected vector is illegal/incomplete, any lexicographic selection has an exact utility key below the document maximum, any artifact pin differs, conflicting cache results exist, or repeated runs produce different hashes.
2. **INSUFFICIENT-CANDIDATE-BREADTH:** any primary document is not support-complete. Report the reason and stop the corpus-level adjudication. For `unsupported-missing-slate`, authorize one separately approved breadth-completion run for exactly those vectors. For `unsupported-no-count-contrast`, require a frozen richer policy-independent slate before more reward calls. Do not run another controller-strength experiment.
3. **NO-OBSERVED-EXACT-OPPORTUNITY:** every support-complete primary document has `G_d = 0`. Exact ties may exist, but the standardized observed exact-optimal sets contain no free count gain over the privacy-blind baseline. Do not proceed to a deployment lexicographic selector.
4. **INSUFFICIENT-PRIMARY-SUPPORT:** at least one primary document has `G_d > 0`, but the one-sided 95% document-bootstrap lower bound for mean `G_d` is zero. Expand the standardized candidate slate only if a separately justified richer policy-independent slate is specified; do not mine adaptive cache paths or run another controller-strength experiment.
5. **ADOPT-EXACT-LEXICOGRAPHIC-COMPOSITION:** every primary document is support-complete, the selector is valid, at least one primary document changes vector, and the one-sided 95% document-bootstrap lower bound for mean `G_d` is greater than zero. Freeze the result artifact, retire additive control as the tie-resolution mechanism, and plan the deployment utility-estimation/search layer against this oracle.

No minimum `Delta P` magnitude is invented. The gate asks whether the expected document-level free count gain is demonstrably positive. Effect size remains fully reported and determines practical priority after the architectural verdict.

### 3.6 Mandatory post-pass boundary

An `ADOPT-EXACT-LEXICOGRAPHIC-COMPOSITION` verdict authorizes an architecture change, not a privacy claim. Before production promotion:

1. Run the held-out LLM re-identification attacker on paired `utility_only_bc_nearest` and `epsilon_zero_lexicographic` outputs for changed primary documents.
2. Measure both `doc_p` attacker success and `out_final` leak-through.
3. Preserve identical task, model, extractor, attacker, and generation settings.
4. Report count gain as diagnostic and attacker success as the privacy outcome.

This paid/rate-limited attacker run requires explicit user approval and is not part of the cache-only gate implementation.

---

## 4. File Map

- Create `src/cloak/ranker/lexicographic.py` — pure exact-utility candidate and selector types; no policy or checkpoint dependency.
- Create `src/cloak/tests/test_lexicographic_ranker.py` — exact-equality, deterministic tie-break, utility dominance, and malformed-input tests.
- Create `scripts/spikes/epsilon_zero_lexicographic_gate.py` — pinned cache loading, candidate validation, archived comparator replay, document-macro aggregation, bootstrap, and artifact writing.
- Create `src/cloak/tests/test_epsilon_zero_lexicographic_gate.py` — cache fixture and report-contract tests for the spike workflow.
- Create `research-wiki/experiments/2026-08-04-epsilon-zero-lexicographic-gate.md` — spec-before-run record and measured adjudication.
- Create after the run `results/ranker_v2/architecture/epsilon-zero-lexicographic-gate.json` — canonical machine-readable local result; `results/` is gitignored, so record its hash in the tracked experiment page rather than forcing it into git.
- Modify after adjudication `docs/specs/RL/ties-by-design.md` — replace stale `0.044` exact-tie language and record the gate outcome.
- Modify after adjudication `docs/specs/RL/interactive-ranker-v2-decision-log.md` — append the closed composition-fork decision.

---

### Task 1: Pre-register the Exact-Set Experiment

**Files:**
- Create: `research-wiki/experiments/2026-08-04-epsilon-zero-lexicographic-gate.md`

**Interfaces:**
- Consumes: this plan, the pinned artifact paths in section 3.1, and the literature pages in section 2.
- Produces: a frozen experiment contract whose hypothesis, validity conditions, outputs, and decision rule match section 3 verbatim.

- [ ] **Step 1: Write the experiment record before writing selector code**

Use frontmatter:

```yaml
---
type: experiment
node_id: exp:epsilon-zero-lexicographic-gate
status: planned
verdict: pending
confidence: pending
created: 2026-08-04
updated: 2026-08-04
tags: [rl, ranker-v2, lexicographic, exact-ties, cache-only]
companion: ../../docs/plans/2026-08-04-epsilon-zero-lexicographic-gate.md
---
```

The record must state:

```markdown
## Hypothesis

Among complete cached vectors with maximum pinned document utility, deterministic
selection of the highest frozen profile-count score yields positive document-macro
count gain outside the four controller-campaign documents with exactly zero measured
utility loss, provided every primary document has complete coverage of the frozen
five-anchor standardized candidate slate.

## Forbidden substitutions

- no 0.044 threshold;
- no structural tie labels;
- no utility-tower logits in admissibility;
- no alpha/gain calibration;
- no per-decision composition;
- no reward calls during the gate.
- no zero-imputation for missing standardized candidates.
```

- [ ] **Step 2: Record exact input hashes**

Run:

```bash
sha256sum \
  results/ranker_v2/environment/ranker-env.json \
  results/ranker_v2/qa/aci-full.utility \
  results/ranker_v2/reward/profile-count-targets.json \
  results/ranker_v2/cache/utility-results.jsonl
```

Copy the four hashes into the experiment record under `## Frozen inputs`.

- [ ] **Step 3: Verify the record contains no stale epsilon**

Run:

```bash
rg -n '0\.044|reader[- ]noise|TIE_EXIT_BOUND' \
  research-wiki/experiments/2026-08-04-epsilon-zero-lexicographic-gate.md
```

Expected: only an explicit statement that these values are forbidden; no threshold use.

- [ ] **Step 4: Commit the pre-registration separately**

```bash
git add research-wiki/experiments/2026-08-04-epsilon-zero-lexicographic-gate.md
git commit -m "docs: preregister epsilon-zero lexicographic gate"
```

---

### Task 2: Implement the Pure Exact Lexicographic Selector

**Files:**
- Create: `src/cloak/ranker/lexicographic.py`
- Create: `src/cloak/tests/test_lexicographic_ranker.py`

**Interfaces:**
- Consumes: pinned component scores and utility weights, ordered action vectors, BC reference vector, and frozen profile-count score.
- Produces: `LexicographicCandidate`, `LexicographicSelection`, `exact_document_utility_key`, `select_utility_only`, and `select_epsilon_zero`.

- [ ] **Step 1: Write failing exact-equality tests**

```python
from decimal import Decimal

from cloak.ranker.lexicographic import LexicographicCandidate, select_epsilon_zero


def candidate(action_id: str, utility: str, privacy: str):
    return LexicographicCandidate(
        doc_id="doc",
        vector_key=(("d", action_id),),
        utility_key=Decimal(utility),
        utility=float(utility),
        privacy_key=Decimal(privacy),
        privacy_score=float(privacy),
        result_hash=f"sha256:{action_id}",
    )


def test_epsilon_zero_excludes_numerically_close_lower_utility():
    exact = candidate("specific", "1.000000000000", "0.1")
    lower = candidate("general", "0.999999999999", "1.0")
    selected = select_epsilon_zero((exact, lower))
    assert selected.selected.vector_key == exact.vector_key
    assert selected.feasible_count == 1


def test_epsilon_zero_maximizes_privacy_only_inside_exact_argmax():
    tied_low = candidate("specific", "1.0", "0.2")
    tied_high = candidate("general", "1.0", "0.8")
    assert select_epsilon_zero((tied_low, tied_high)).selected == tied_high
```

Keep the `candidate` helper inside the test module. Production selection must not accept a numeric epsilon argument.

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest \
  src/cloak/tests/test_lexicographic_ranker.py -q
```

Expected: import failure because `cloak.ranker.lexicographic` does not exist.

- [ ] **Step 3: Implement immutable candidate and selection records**

Use this public shape:

```python
@dataclass(frozen=True)
class LexicographicCandidate:
    doc_id: str
    vector_key: tuple[tuple[str, str], ...]
    utility_key: Decimal
    utility: float
    privacy_key: Decimal
    privacy_score: float
    result_hash: str


@dataclass(frozen=True)
class LexicographicSelection:
    selected: LexicographicCandidate
    feasible_count: int
    optimal_utility_key: Decimal
    feasible_privacy_min: Decimal
    feasible_privacy_max: Decimal
```

Validate non-empty IDs/vector keys, finite float summaries, one action per decision, and non-empty candidate collections. Do not import PyTorch.

- [ ] **Step 4: Implement exact cached utility keys**

Use the existing `utility_binding` contract and compare the weighted numerator, since the denominator is constant within a document:

```python
def exact_document_utility_key(
    component_scores: Mapping[str, float],
    utility_artifact: Mapping,
    doc_id: str,
) -> Decimal:
    binding = utility_binding(utility_artifact, doc_id)
    return sum(
        Decimal(str(binding["weights"][assertion_id]))
        * Decimal(str(component_scores[assertion_id]))
        for assertion_id in sorted(binding["weights"])
    )
```

Raise when any weighted policy-role score is missing. Extra non-policy component scores are valid and remain ignored by this key, matching `document_utility`. This key exists only to compare candidates from the same document; do not compare it across documents.

- [ ] **Step 5: Implement both deterministic selectors**

```python
def select_epsilon_zero(
    candidates: Sequence[LexicographicCandidate],
) -> LexicographicSelection:
    optimum = max(candidate.utility_key for candidate in candidates)
    feasible = tuple(c for c in candidates if c.utility_key == optimum)
    selected = min(feasible, key=lambda c: (-c.privacy_key, c.vector_key))
    return _selection(selected, feasible, optimum)


def select_utility_only(
    candidates: Sequence[LexicographicCandidate],
    bc_vector_key: tuple[tuple[str, str], ...],
) -> LexicographicSelection:
    optimum = max(candidate.utility_key for candidate in candidates)
    feasible = tuple(c for c in candidates if c.utility_key == optimum)
    selected = min(
        feasible,
        key=lambda c: (_hamming(c.vector_key, bc_vector_key), c.vector_key),
    )
    return _selection(selected, feasible, optimum)
```

Do not pass privacy into `select_utility_only`. Do not use input order as a tie-break.

- [ ] **Step 6: Add malformed-input and permutation-invariance tests**

Cover:

- empty candidates;
- candidates from two document IDs;
- duplicate decision IDs in a vector;
- BC vector with a different decision ordering;
- equal utility and privacy resolved by stable vector key;
- selection unchanged under 100 seeded input permutations;
- exact utility key distinguishes values that differ below `1e-9`.

- [ ] **Step 7: Run focused tests**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest \
  src/cloak/tests/test_lexicographic_ranker.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit selector core**

```bash
git add src/cloak/ranker/lexicographic.py \
        src/cloak/tests/test_lexicographic_ranker.py
git commit -m "feat: add exact lexicographic selector"
```

---

### Task 3: Build the Pinned Candidate Corpus

**Files:**
- Create: `scripts/spikes/epsilon_zero_lexicographic_gate.py`
- Create: `src/cloak/tests/test_epsilon_zero_lexicographic_gate.py`

**Interfaces:**
- Consumes: `UtilityCache.entries`, `RankerDocument`, `ProfileCountTargets`, utility artifact, environment and artifact hashes.
- Produces: `dict[str, CandidateCorpusDocument]` containing the expected standardized slate, validated cached candidates, support status, and an explicit rejection audit.

- [ ] **Step 1: Write a failing cache-corpus fixture test**

Construct two documents with:

- two complete base vectors;
- five deterministic anchor sources that deduplicate to at least two count-distinguishable expected vectors;
- one `reader_refresh=true` duplicate;
- one incomplete vector;
- one vector with a mismatched environment hash;
- one illegal injectivity collision.

Assert:

```python
corpus, audit = load_candidate_corpus(...)
assert len(corpus["doc-valid"].gate_candidates) == 2
assert audit["reader_refresh_excluded"] == 1
assert audit["incomplete_vector_excluded"] == 1
assert audit["pin_mismatch_excluded"] == 1
assert audit["illegal_vector_excluded"] == 1
```

Add separate fixtures proving:

- one missing expected anchor yields `candidate_support_status == "unsupported-missing-slate"`;
- a complete slate with no count contrast yields `candidate_support_status == "unsupported-no-count-contrast"`;
- extra adaptively cached vectors do not repair a missing standardized anchor;
- source aliases from exact-deduplicated anchors are all preserved.

- [ ] **Step 2: Run the fixture test and verify it fails**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest \
  src/cloak/tests/test_epsilon_zero_lexicographic_gate.py \
  -k candidate_corpus -q
```

Expected: import failure because the spike module does not exist.

- [ ] **Step 3: Build the standardized slate before reading outcomes**

Call `build_anchor_trajectories(document, count_reward)` for every retained document before joining any utility result. Preserve each trajectory's merged `sources` and `ordered_action_vector`; fail if an anchor vector is incomplete or illegal. Do not reimplement the five chooser functions in the spike.

Represent the expected slate with an immutable record:

```python
@dataclass(frozen=True)
class ExpectedSlateVector:
    vector_key: tuple[tuple[str, str], ...]
    sources: tuple[str, ...]
    privacy_key: Decimal


@dataclass(frozen=True)
class CandidateCorpusDocument:
    doc_id: str
    expected_slate: tuple[ExpectedSlateVector, ...]
    gate_candidates: tuple[LexicographicCandidate, ...]
    expanded_cache_candidates: tuple[LexicographicCandidate, ...]
    missing_expected_vectors: tuple[ExpectedSlateVector, ...]
    candidate_support_status: Literal[
        "unsupported-missing-slate",
        "unsupported-no-count-contrast",
        "support-complete",
    ]
```

The support join must distinguish an absent cache result from an excluded invalid result. Both produce `unsupported-missing-slate`, while the rejection audit retains the reason.

- [ ] **Step 4: Implement strict cache loading**

The loader must iterate validated `UtilityCache.entries`, retain only base rows, verify pins, and preserve `identity["ordered_action_vector"]` as the canonical order. Apply the same `_demote_out_of_scope_decisions` and `_drop_zero_signal_documents` scope used by the production trainer before validating complete vectors. Never parse JSONL ad hoc while bypassing `UtilityCache` validation.

Use `document_utility` to verify the stored float and `exact_document_utility_key` for admissibility. Compute the document count score over the active document decisions, matching the production diagnostic:

```python
score_keys = [
    Decimal(str(float(targets.action_scores(
        decision.decision_id, (action_vector[decision.decision_id],)
    )[0])))
    for decision in document.policy_decisions
]
privacy_key = sum(score_keys) / Decimal(len(score_keys))
privacy_score = float(privacy_key)
```

Use `privacy_key` for deterministic ordering and `privacy_score` only for JSON reporting. Do not call `ProfileCountTargets.selected_document_score`; it expects the complete artifact-wide decision set rather than one document vector.

- [ ] **Step 5: Implement policy-independent legality replay**

Walk each document's decisions in first-occurrence order using existing environment helpers:

```python
reserved = tuple(_fixed_fill_claims(document, occurrence_by_id))
claimed = {}
for decision_id, action_id in vector_key:
    decision = decision_by_id[decision_id]
    legal = legal_action_ids(decision, claimed, reserved)
    if action_id not in legal:
        return False
    action = _action_by_id(decision, action_id)
    if action.mode == "level":
        claimed.setdefault(_fill_key(action.fill), decision.decision_id)
return True
```

Fail closed if vector decision order differs from the environment.

- [ ] **Step 6: Reject conflicting base results**

If two base cache identities with identical `(doc_id, ordered_action_vector, environment_hash, utility_artifact_hash)` contain different result hashes or component scores, raise `ValueError`. Do not average reader results; exact-set membership cannot survive an unresolved duplicate contract.

- [ ] **Step 7: Classify candidate support without zero imputation**

After joining the expected slate to validated candidates:

```python
if missing_expected_vectors:
    candidate_support_status = "unsupported-missing-slate"
elif len({row.privacy_key for row in gate_candidates}) < 2:
    candidate_support_status = "unsupported-no-count-contrast"
else:
    candidate_support_status = "support-complete"
```

Do not compute `G_d` in this task. Store `missing_expected_vectors`, `expected_slate_size`, `cached_slate_size`, and `candidate_support_status`. Put only standardized joined vectors in `gate_candidates`; put all other validated vectors in `expanded_cache_candidates`. Add a test proving an extra high-privacy exact-utility vector changes only the expanded-cache diagnostic and cannot change the primary `G_d` or verdict.

- [ ] **Step 8: Run cache-corpus tests**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest \
  src/cloak/tests/test_epsilon_zero_lexicographic_gate.py \
  -k candidate_corpus -q
```

Expected: all candidate-corpus tests pass.

- [ ] **Step 9: Commit candidate-corpus implementation**

```bash
git add scripts/spikes/epsilon_zero_lexicographic_gate.py \
        src/cloak/tests/test_epsilon_zero_lexicographic_gate.py
git commit -m "test: build pinned lexicographic candidate corpus"
```

---

### Task 4: Implement Document-Level Adjudication

**Files:**
- Modify: `scripts/spikes/epsilon_zero_lexicographic_gate.py`
- Modify: `src/cloak/tests/test_epsilon_zero_lexicographic_gate.py`

**Interfaces:**
- Consumes: validated candidate corpus and BC trajectories.
- Produces: one support-aware per-document gate record and document-macro summary with a closed verdict enum.

- [ ] **Step 1: Write failing per-document selection tests**

Create fixture documents representing:

1. complete standardized support, one utility optimum, and no tie;
2. complete standardized support and two exact utility optima with equal privacy;
3. complete standardized support and two exact utility optima with different privacy;
4. complete standardized support and a more-private vector with utility lower by `1e-12`;
5. complete standardized support and multiple exact optima where the BC-nearest and privacy-max vectors differ;
6. one missing standardized vector despite several extra cached vectors;
7. a one-vector document.

Assert that only cases 3 and 5 have positive `G_d`, case 4 excludes the more-private vector, and cases 6 and 7 return `G_d: null` with an unsupported status.

- [ ] **Step 2: Implement `evaluate_document`**

Use this signature:

```python
def evaluate_document(
    document: RankerDocument,
    candidate_document: CandidateCorpusDocument,
    bc_vector_key: tuple[tuple[str, str], ...],
    *,
    population: Literal["primary", "campaign"],
) -> dict[str, Any]:
    ...
```

For unsupported documents, return coverage fields with `opportunity_status: null`; set selector records and `free_count_gain` to `null`. For support-complete documents, run both primary selectors over `candidate_document.gate_candidates` only, set `opportunity_status` to `supported-opportunity` when `G_d > 0` and `supported-no-opportunity` otherwise, and enforce:

```python
assert lex.utility_key == utility_only.utility_key == max_utility_key
assert lex.privacy_key >= utility_only.privacy_key
assert free_count_gain >= 0
```

Evaluate `expanded_cache_candidates` in a separately named diagnostic block. Add a test proving no expanded-cache field is read by verdict construction.

- [ ] **Step 3: Implement document bootstrap**

Use deterministic percentile bootstrap over document-level `G_d`, including zeros:

```python
def lower_mean_gain_bound(
    gains: Sequence[float], *, seed: int = 20260804, samples: int = 10_000
) -> float:
    ...
```

Return the one-sided 95% lower bound at the 5th percentile. Raise on an empty primary population. Report the observed mean, lower bound, sample count, bootstrap seed, and resample count.

Pass only non-null gains from support-complete primary documents. Add a test proving that adding an unsupported document cannot change the estimate and is reported through support coverage instead.

- [ ] **Step 4: Implement the closed verdict enum**

Allowed values only:

```python
class GateVerdict(str, Enum):
    INVALID = "invalid"
    INSUFFICIENT_CANDIDATE_BREADTH = "insufficient-candidate-breadth"
    NO_OBSERVED_EXACT_OPPORTUNITY = "no-observed-exact-opportunity"
    INSUFFICIENT_PRIMARY_SUPPORT = "insufficient-primary-support"
    ADOPT_EXACT_LEXICOGRAPHIC_COMPOSITION = (
        "adopt-exact-lexicographic-composition"
    )
```

Implement section 3.5 in exactly that order. Store every boolean predicate used to choose the verdict in `adjudication_checks`, including `all_primary_documents_support_complete`. Add a test where positive subset gains still yield `INSUFFICIENT_CANDIDATE_BREADTH` because one primary document lacks one standardized vector.

- [ ] **Step 5: Add order-invariance and repeatability tests**

Run the complete synthetic gate twice with shuffled cache-record order. Require byte-identical canonical JSON after removing only an optional wall-time field. Do not write timestamps into the canonical result.

- [ ] **Step 6: Run adjudication tests**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest \
  src/cloak/tests/test_lexicographic_ranker.py \
  src/cloak/tests/test_epsilon_zero_lexicographic_gate.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit adjudication logic**

```bash
git add scripts/spikes/epsilon_zero_lexicographic_gate.py \
        src/cloak/tests/test_epsilon_zero_lexicographic_gate.py
git commit -m "test: adjudicate epsilon-zero lexicographic selector"
```

---

### Task 5: Add Archived Additive-Controller Comparators

**Files:**
- Modify: `scripts/spikes/epsilon_zero_lexicographic_gate.py`
- Modify: `src/cloak/tests/test_epsilon_zero_lexicographic_gate.py`

**Interfaces:**
- Consumes: repeated `--checkpoint LABEL=PATH` arguments, lambda menu, semantic representation manifest, candidate-corpus hash join.
- Produces: report-only per-checkpoint greedy lambda-zero/lambda-max comparisons without influencing the lexicographic verdict.

- [ ] **Step 1: Write a failing comparator cache-hit test**

Use a tiny deterministic policy fixture whose lambda-zero and lambda-max vectors are both in the synthetic candidate corpus. Assert that the comparator reports:

```python
{
    "utility_gap_to_exact_optimum": ...,
    "privacy_gap_to_lexicographic": ...,
    "inside_exact_optimal_set": ...,
    "chooses_privacy_max_inside_exact_set": ...,
}
```

- [ ] **Step 2: Implement explicit checkpoint arguments**

CLI form:

```bash
--checkpoint detached=results/ranker_v2/architecture/count_to_gain/detached-s47.pt \
--checkpoint coupled=results/ranker_v2/architecture/count_to_gain/coupled-s47.pt
```

Never scan the results tree and silently choose checkpoints. Hash every checkpoint and store its architecture/config metadata in the report.

- [ ] **Step 3: Replay greedy trajectories on every retained document**

Build the policy over the full retained document set so runtime-type indices match training. Use the production sequential menu, claimed-fill, state-advance, and lambda-profile paths. Evaluate lambda zero and lambda max only.

Join vectors to the validated cache by `(doc_id, vector_key)`. A cache miss is a report row with `status: cache-miss`; do not dispatch a utility request.

- [ ] **Step 4: Keep comparator metrics out of the verdict**

Add a test that changes archived comparator outputs while holding the candidate corpus fixed and proves the gate verdict and lexicographic hashes remain unchanged.

- [ ] **Step 5: Run comparator tests**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest \
  src/cloak/tests/test_epsilon_zero_lexicographic_gate.py \
  -k additive_comparator -q
```

Expected: all comparator tests pass.

- [ ] **Step 6: Commit comparator support**

```bash
git add scripts/spikes/epsilon_zero_lexicographic_gate.py \
        src/cloak/tests/test_epsilon_zero_lexicographic_gate.py
git commit -m "test: compare exact selector with additive checkpoints"
```

---

### Task 6: Run a Representative Real-Data Preflight

**Files:**
- Modify if defects are found: `scripts/spikes/epsilon_zero_lexicographic_gate.py`
- Modify if defects are found: `src/cloak/tests/test_epsilon_zero_lexicographic_gate.py`

**Interfaces:**
- Consumes: one campaign document, two support-complete non-campaign documents, real pinned artifacts and cache.
- Produces: a preflight JSON printed to stdout only; no canonical result yet.

- [ ] **Step 1: Add deterministic preflight scope support**

Support repeatable `--doc-id` arguments and `--preflight-primary-docs N`. The latter selects the first `N` sorted non-campaign document IDs whose standardized slate is support-complete, before inspecting utility or privacy gains. Scope must affect only evaluation population, not artifact validation. Also print the first five unsupported primary document IDs and missing anchor sources so the preflight exercises both branches.

- [ ] **Step 2: Run the three-document preflight**

Choose `aci/D2N005` plus the first two sorted support-complete non-campaign documents. The script performs this selection before computing gains. If fewer than two exist, stop with `INSUFFICIENT-CANDIDATE-BREADTH`; do not substitute campaign documents or adaptively sampled high-coverage documents.

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -u \
  scripts/spikes/epsilon_zero_lexicographic_gate.py \
  --environment results/ranker_v2/environment/ranker-env.json \
  --utility-artifact results/ranker_v2/qa/aci-full.utility \
  --profile-count-targets results/ranker_v2/reward/profile-count-targets.json \
  --utility-cache results/ranker_v2/cache/utility-results.jsonl \
  --checkpoint detached=results/ranker_v2/architecture/count_to_gain/detached-s47.pt \
  --checkpoint coupled=results/ranker_v2/architecture/count_to_gain/coupled-s47.pt \
  --doc-id aci/D2N005 \
  --preflight-primary-docs 2 \
  --stdout
```

- [ ] **Step 3: Inspect actual selected vectors**

For each preflight document, manually verify:

- both selectors' action vectors exist in the cache;
- every expected standardized anchor is present for support-complete documents;
- unsupported examples carry `G_d: null` rather than zero;
- utilities recomputed from component scores match;
- a positive `G_d` comes only from exact-optimal candidates;
- the privacy-max choice corresponds to higher frozen profile-count scores;
- no structural dependency label was read.

Record the three document IDs and selected vector hashes in the experiment record under `## Real-data preflight`.

- [ ] **Step 4: Run all focused tests after any correction**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest \
  src/cloak/tests/test_lexicographic_ranker.py \
  src/cloak/tests/test_epsilon_zero_lexicographic_gate.py \
  src/cloak/tests/test_interactive_ranker.py \
  src/cloak/tests/test_semantic_ranker.py -q
```

Expected: all tests pass.

---

### Task 7: Run the Full Cache-Only Gate

**Files:**
- Create: `results/ranker_v2/architecture/epsilon-zero-lexicographic-gate.json`
- Modify: `research-wiki/experiments/2026-08-04-epsilon-zero-lexicographic-gate.md`

**Interfaces:**
- Consumes: all retained documents, standardized five-anchor slates, validated base cache, frozen count targets, explicit archived checkpoints.
- Produces: canonical result artifact and measured experiment verdict.

- [ ] **Step 1: Run the full gate once**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -u \
  scripts/spikes/epsilon_zero_lexicographic_gate.py \
  --environment results/ranker_v2/environment/ranker-env.json \
  --utility-artifact results/ranker_v2/qa/aci-full.utility \
  --profile-count-targets results/ranker_v2/reward/profile-count-targets.json \
  --utility-cache results/ranker_v2/cache/utility-results.jsonl \
  --checkpoint detached=results/ranker_v2/architecture/count_to_gain/detached-s47.pt \
  --checkpoint coupled=results/ranker_v2/architecture/count_to_gain/coupled-s47.pt \
  --output results/ranker_v2/architecture/epsilon-zero-lexicographic-gate.json
```

This is CPU/cache-only and should complete in minutes. It must report zero remote tasks and zero reader work items.

- [ ] **Step 2: Run it a second time into a temporary file**

```bash
PYTHONPATH=src:scripts .venv/bin/python -u \
  scripts/spikes/epsilon_zero_lexicographic_gate.py \
  --environment results/ranker_v2/environment/ranker-env.json \
  --utility-artifact results/ranker_v2/qa/aci-full.utility \
  --profile-count-targets results/ranker_v2/reward/profile-count-targets.json \
  --utility-cache results/ranker_v2/cache/utility-results.jsonl \
  --checkpoint detached=results/ranker_v2/architecture/count_to_gain/detached-s47.pt \
  --checkpoint coupled=results/ranker_v2/architecture/count_to_gain/coupled-s47.pt \
  --output /tmp/epsilon-zero-lexicographic-gate-repeat.json
cmp results/ranker_v2/architecture/epsilon-zero-lexicographic-gate.json \
    /tmp/epsilon-zero-lexicographic-gate-repeat.json
```

Expected: `cmp` exits zero.

- [ ] **Step 3: Validate the canonical artifact**

Run:

```bash
jq -e '
  .epsilon == "0" and
  .remote_tasks == 0 and
  .reader_work_items == 0 and
  .summary.primary.document_count > 0 and
  ([.documents[] | select(
      .population == "primary" and
      .candidate_support_status == "support-complete"
    ) |
    .epsilon_zero_lexicographic.utility_key == .utility_only.utility_key] | all)
' results/ranker_v2/architecture/epsilon-zero-lexicographic-gate.json
```

Expected: `true` and exit zero.

- [ ] **Step 4: Fill the experiment results without interpretation drift**

Set experiment `status: done` and record:

- exact verdict string;
- primary/campaign document counts;
- standardized-slate support-completeness fraction, unsupported counts by reason, and every missing anchor vector;
- fraction with positive `G_d`;
- mean, median, and lower bound for `G_d` over support-complete documents only;
- standardized-slate coverage distribution and non-adjudicating expanded-cache coverage;
- additive comparator cache-hit count and both failure rates;
- no-reward-call confirmation;
- limitations: frozen five-anchor candidate slate rather than exhaustive legal search, count proxy, no deployment utility estimator, no attacker result.

Do not write “privacy improved” unless the attacker leg has run. Write “profile-count score improved at exact measured utility.”

- [ ] **Step 5: Hash the local result and commit code, tests, and record**

Run and copy the emitted hash into the experiment record:

```bash
sha256sum results/ranker_v2/architecture/epsilon-zero-lexicographic-gate.json
```

```bash
git add src/cloak/ranker/lexicographic.py \
        src/cloak/tests/test_lexicographic_ranker.py \
        src/cloak/tests/test_epsilon_zero_lexicographic_gate.py \
        scripts/spikes/epsilon_zero_lexicographic_gate.py \
        research-wiki/experiments/2026-08-04-epsilon-zero-lexicographic-gate.md
git commit -m "validate: adjudicate exact lexicographic composition"
```

---

### Task 8: Close the Composition Fork

**Files:**
- Modify: `docs/specs/RL/ties-by-design.md`
- Modify: `docs/specs/RL/interactive-ranker-v2-decision-log.md`

**Interfaces:**
- Consumes: canonical gate artifact and completed experiment record.
- Produces: one normative architecture decision and explicit deletion/next-step consequences.

- [ ] **Step 1: Correct stale `0.044` semantics regardless of verdict**

In `docs/specs/RL/ties-by-design.md`:

- define exact tie solely by exact pinned utility equality plus serialization-safe implementation;
- remove claims that `0.044` is an intrinsic reader floor or valid equivalence budget;
- distinguish measurement uncertainty, statistical equivalence, and user policy budget;
- state that the epsilon-zero gate uses none of them.

- [ ] **Step 2: Record the verdict's forced consequence**

If the verdict is `adopt-exact-lexicographic-composition`, record:

- additive `alpha * P` is rejected as the tie-resolution solution concept;
- no more gain-head, alpha-strength, softcap, or tie-hinge experiment may be justified as solving exact tie ownership;
- the next architecture task is a utility estimator/search method evaluated against the exact-set oracle;
- lambda zero remains exact identity; positive lambdas share the epsilon-zero selector until a separately justified document budget exists.

If the verdict is `insufficient-candidate-breadth`, record that the cache-only gate cannot make a corpus-level architecture decision. For missing slate vectors, authorize one separately approved reward run containing exactly those vectors and preserve the existing slate definition. For standardized slates with no count contrast, require a frozen richer policy-independent slate before authorizing reward calls. Prohibit controller or adaptive-policy expansion before rerunning the gate.

If the verdict is `no-observed-exact-opportunity`, record that exact-set selection is formally correct but not practically load-bearing in the standardized observed pool.

If the verdict is `insufficient-primary-support`, authorize a separately specified richer policy-independent slate only if the observed effect size justifies it; prohibit adaptive cache mining and another controller intervention before rerunning this unchanged gate.

- [ ] **Step 3: Append the decision-log entry**

Include:

- frozen inputs and artifact hash;
- exact verdict and all adjudication predicates;
- literature-backed solution concept;
- claims supported and unsupported;
- next authorized implementation step;
- mechanisms explicitly retired or retained.

- [ ] **Step 4: Run documentation consistency checks**

Run:

```bash
rg -n '0\.044|reader[- ]noise floor|TIE_EXIT_BOUND' \
  docs/specs/RL/ties-by-design.md \
  docs/specs/RL/interactive-ranker-v2-decision-log.md
```

Inspect every remaining match. Historical entries may retain measured values only when clearly marked historical/stale; normative text must not use `0.044` as epsilon.

- [ ] **Step 5: Run final verification**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest \
  src/cloak/tests/test_lexicographic_ranker.py \
  src/cloak/tests/test_epsilon_zero_lexicographic_gate.py \
  src/cloak/tests/test_interactive_ranker.py \
  src/cloak/tests/test_semantic_ranker.py -q

git diff --check
```

Expected: all tests pass and `git diff --check` prints nothing.

- [ ] **Step 6: Commit the closed fork**

```bash
git add docs/specs/RL/ties-by-design.md \
        docs/specs/RL/interactive-ranker-v2-decision-log.md
git commit -m "docs: close exact lexicographic composition fork"
```

---

## 5. Failure Interpretation Matrix

| Finding | Interpretation | Next action |
|---|---|---|
| Exact selector loses utility | Implementation/pin defect; impossible under the definition | Stop and debug; no architectural inference |
| Any primary document lacks a standardized vector, has only one unique standardized vector, or has no standardized count contrast | Candidate breadth is missing; this is not evidence of zero opportunity | Return `INSUFFICIENT-CANDIDATE-BREADTH`; score missing frozen vectors, or freeze a richer policy-independent slate when count contrast is absent |
| Every support-complete primary document has `G_d = 0` and all primary documents are support-complete | Standardized exact optimal sets offer no observed free count opportunity | Do not build deployment lexicographic machinery |
| Some positive gains, bootstrap lower bound is zero | Opportunity exists but current breadth cannot establish corpus-level leverage | Expand candidate breadth once across non-campaign documents |
| Positive lower bound, additive comparator often below `U*` | Exact tie opportunity is load-bearing and additive composition also spends utility | Adopt exact lexicographic composition; prioritize deployment utility estimation |
| Positive lower bound, additive comparator reaches `U*` but misses max `P` | Additive controller fails tie ownership even when utility is preserved | Adopt exact lexicographic composition; controller-strength work is closed |
| Count improves but attacker later does not | Count proxy is not a valid privacy objective for this operating region | Keep the selector mechanics separate; revisit privacy scoring before promotion |
| Oracle works but learned deployment selector fails | Composition is correct; utility estimation/search is the remaining defect | Improve the utility model or allow inference-time search; do not return to alpha tuning |

## 6. Why This Is Not Another Measurement Loop

This gate terminates a design fork:

- It executes the lexicographic operator directly instead of predicting equivalence.
- It uses complete end-to-end utility vectors instead of structural labels.
- It requires the same deterministic candidate slate on every primary document instead of mistaking adaptively sampled cache coverage for a population.
- It cannot be defeated by tower-logit scale, 41% tower ordering error, dead gain features, or weak alpha gradients because none are inputs.
- It enforces zero utility loss by construction rather than testing whether an additive controller happens to approximate it.
- Its result has a forced engineering consequence: either complete the frozen candidate slate once, retire additive tie control, justify one richer policy-independent slate, or abandon exact ties as a meaningful privacy lever.

The remaining deployment question is intentionally exposed rather than hidden: a production policy still needs a way to approximate or search the exact measured-utility argmax set on unseen documents. That is the next model-design problem only if this gate proves the target set contains useful count variation.
