"""Recall-first semantic escalation for the relation opportunity miner.

Passed as `escalator` to `relation_support_opportunities`, this is consulted ONLY on a cue-miss and
can only return accept — so it recovers the miner's false-negatives (e.g. dialogue notes that the
lexical cue lexicon scores at 0 opportunities) without ever regressing the cue-matched set.

Two accept-biased tiers (objective: drive false-negatives toward 0, tolerate false positives —
these gate teacher *targets*, and the reader three-point gate is still the real acceptance check):

  1. MedNLI (optional, accept-only): a high-confidence clinical-entailment score accepts cheaply and
     skips the LLM; anything below defers. It never rejects, so it cannot cause a false-negative.
  2. MedGemma-4b judge: grounded, coreference-aware. Accepts unless the excerpt clearly does not
     assert the relation. On any ambiguity or parse failure it accepts (recall-first).

Premise = the anchor's surgical clause-stitch (both arguments' clauses, middle elided). If that
proves to truncate evidence and cost recall, widen it here — the callback receives `document`.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence

# Natural-language claim per relation, filled with the SOURCE surfaces (not generalized levels — the
# judge decides whether the *source* asserts the relation, independent of the privacy render).
RELATION_CLAIM = {
    "prescribed_with": "{s} is a medication prescribed to treat {o}",
    "procedure_for": "{s} is a procedure or treatment given for {o}",
    "tests_for": "{s} is a test or scan done to investigate {o}",
    "contraindicated_because_of": "the patient must NOT be given {s} because they have {o}",
    # subject is the CAUSE, object the effect (contract: condition -> condition/symptom), so the
    # claim is active. The passive form ("{s} is caused or explained by {o}") inverted the direction
    # and made the judge assess the wrong proposition.
    "causes_or_explains": "{s} causes or explains {o}",
}
# For these relations the claim template names the drug/test/procedure FIRST, but the miner's subject
# is the CONDITION (object is the drug/test). So _claim swaps subject<->object for them; the other two
# (contraindicated_because_of, causes_or_explains) are already subject-first and need no swap.
_OBJECT_FIRST = {"prescribed_with", "procedure_for", "tests_for"}

# The judge system prompt is assembled PER RELATION: a shared grounding preamble + only the target
# relation's specific rule (if any) + only that relation's worked examples. This isolation is
# deliberate — a single shared prompt made the judge non-modular: adding the causal rule measurably
# drifted prescribed/tests verdicts (+10 opportunities on D2N002). Per-relation prompts trade the
# constant-prefix cache benefit for edit isolation (a per-relation tweak can't spill onto others).
_JUDGE_PREAMBLE = (
    "You are a careful clinical NLP annotator. Given a short excerpt from a doctor-patient visit "
    "transcript and a candidate clinical relation, decide ONLY whether THIS excerpt actually states "
    "that relation FOR THIS PATIENT.\n"
    "General rules:\n"
    "1. The excerpt must actually assert the candidate relation between the two named items. Mere "
    "co-occurrence in the excerpt is NOT enough.\n"
    "2. A visit discusses several problems in sequence. Watch for topic switches ('and then', 'in "
    "terms of your X', 'for your Y', 'as for', a new problem name). An item belonging to one problem "
    "must NOT be attributed to a different problem mentioned nearby.\n"
    "3. Do not use general medical plausibility. A pairing that is medically plausible but not "
    "asserted in THIS excerpt is false.\n"
    "4. Resolve pronouns ('you', 'it') to the problem actually under discussion at that point."
)
# Extra rule for relations that need one beyond the general grounding rules (others: none).
_RELATION_RULES = {
    "contraindicated_because_of": (
        "Relation rule: a CONTRAINDICATION claim ('must NOT be given X because they have Y') is TRUE "
        "only if the excerpt says X is avoided, held, stopped, not started, or unsafe BECAUSE of Y. "
        "If the excerpt says the patient takes / continues / is treated with X (for any problem), or "
        "merely mentions X and Y near each other, that is NOT a contraindication — answer false."
    ),
    "causes_or_explains": (
        "Relation rule: a CAUSAL claim ('X causes or explains Y') is TRUE only if the excerpt "
        "EXPLICITLY attributes Y to X — states that X causes, explains, underlies, or is an "
        "exacerbation/manifestation of Y, in that direction. Two conditions merely mentioned in the "
        "same visit are NOT a causal link, and a normal, absent, or denied finding ('no chest pain', "
        "'denies dizziness', 'a normal blood sugar') can never be a cause or an effect — answer false. "
        "Check the direction: 'X causes Y' is false if the excerpt only supports 'Y causes X'."
    ),
}
# Only the target relation's own worked examples. ENTITIES ARE DELIBERATELY OFF-CORPUS (migraine,
# cataract, pneumonia, peptic ulcer, gout, cellulitis ...) so no example overlaps an ACI test
# document -- an example built from the doc under test would let the judge parrot the answer instead
# of reasoning, invalidating any validation. Never seed these with an entity from a test note.
_RELATION_EXAMPLES = {
    "prescribed_with": (
        "EXCERPT: \"[doctor] for your migraine , i'm going to start you on sumatriptan .\"\n"
        "CANDIDATE: sumatriptan is a medication prescribed to treat migraine\n"
        "{\"asserted\": true, \"why\": \"explicitly started for the migraine\"}\n"
        "EXCERPT: \"[doctor] we'll treat the migraine with sumatriptan . and then for your reflux , take omeprazole .\"\n"
        "CANDIDATE: omeprazole is a medication prescribed to treat migraine\n"
        "{\"asserted\": false, \"why\": \"omeprazole is for the reflux, a separate problem\"}"
    ),
    "tests_for": (
        "EXCERPT: \"[doctor] to keep an eye on your asthma , i'm ordering spirometry .\"\n"
        "CANDIDATE: spirometry is a test done to investigate asthma\n"
        "{\"asserted\": true, \"why\": \"ordered to monitor the asthma\"}\n"
        "EXCERPT: \"[doctor] let's get a chest x-ray for your cough . and for your gout , continue allopurinol .\"\n"
        "CANDIDATE: a chest x-ray is a test done to investigate gout\n"
        "{\"asserted\": false, \"why\": \"x-ray is for the cough; gout is a separate item\"}"
    ),
    "procedure_for": (
        "EXCERPT: \"[doctor] for your cataract , we'll schedule phacoemulsification .\"\n"
        "CANDIDATE: phacoemulsification is a procedure given for cataract\n"
        "{\"asserted\": true, \"why\": \"scheduled to treat the cataract\"}\n"
        "EXCERPT: \"[patient] i take albuterol for my asthma . [doctor] ok , and i'll drain that abscess .\"\n"
        "CANDIDATE: abscess drainage is a procedure given for asthma\n"
        "{\"asserted\": false, \"why\": \"drainage is for the abscess; asthma is separate\"}"
    ),
    "contraindicated_because_of": (
        "EXCERPT: \"[doctor] given your peptic ulcer we need to avoid nsaids like ibuprofen .\"\n"
        "CANDIDATE: the patient must NOT be given ibuprofen because they have peptic ulcer\n"
        "{\"asserted\": true, \"why\": \"explicitly avoiding ibuprofen due to the ulcer\"}\n"
        "EXCERPT: \"[patient] i take allopurinol for my gout .\"\n"
        "CANDIDATE: the patient must NOT be given allopurinol because they have gout\n"
        "{\"asserted\": false, \"why\": \"taken to treat the gout; not contraindicated\"}"
    ),
    "causes_or_explains": (
        "EXCERPT: \"[doctor] your shortness of breath is due to your pneumonia .\"\n"
        "CANDIDATE: pneumonia causes or explains the shortness of breath\n"
        "{\"asserted\": true, \"why\": \"breathlessness attributed to the pneumonia\"}\n"
        "EXCERPT: \"[doctor] your shortness of breath is due to your pneumonia .\"\n"
        "CANDIDATE: the shortness of breath causes or explains the pneumonia\n"
        "{\"asserted\": false, \"why\": \"pneumonia explains the breathlessness, not the reverse\"}\n"
        "EXCERPT: \"[doctor] you do n't have any chest pain , good . this leg swelling is from your cellulitis .\"\n"
        "CANDIDATE: chest pain causes or explains the cellulitis\n"
        "{\"asserted\": false, \"why\": \"no chest pain present; no causal link stated\"}\n"
        "EXCERPT: \"[doctor] your history includes eczema and asthma .\"\n"
        "CANDIDATE: eczema causes or explains the asthma\n"
        "{\"asserted\": false, \"why\": \"both listed in history; no causation stated\"}"
    ),
}
_JUDGE_ANSWER_FORMAT = "Answer strictly as JSON: {\"asserted\": true|false, \"why\": \"<=12 words\"}."


def _judge_system(relation: str | None) -> str:
    """Assemble the judge system prompt for ONE relation: shared preamble + that relation's rule (if
    any) + its examples + the answer format. relation=None -> preamble + format only (no relation
    context); production always passes a relation."""
    parts = [_JUDGE_PREAMBLE]
    rule = _RELATION_RULES.get(relation or "")
    if rule:
        parts.append(rule)
    parts.append(_JUDGE_ANSWER_FORMAT)
    examples = _RELATION_EXAMPLES.get(relation or "")
    if examples:
        parts.append("Examples:\n" + examples)
    return "\n".join(parts)


def _claim(relation: str, subject: str, object_: str) -> str | None:
    template = RELATION_CLAIM.get(relation)
    if template is None:
        return None
    if relation in _OBJECT_FIRST:
        # template names the drug/test/procedure first; the miner's subject is the condition.
        subject, object_ = object_, subject
    return template.format(s=subject, o=object_)


def _judge_user_msg(premise: str, claim: str) -> str:
    return (f"EXCERPT:\n{premise}\n\nCANDIDATE RELATION: {claim}.\n\n"
            f"Does the excerpt state that, for this patient, {claim}? Answer JSON only.")


def _subject_object_surfaces(arguments: Sequence[Mapping]) -> tuple[str | None, str | None]:
    sub = next((a for a in arguments if a.get("role") == "subject"), None)
    obj = next((a for a in arguments if a.get("role") == "object"), None)

    def surf(a):
        if a is None:
            return None
        s = str(a.get("surface") or a.get("literal") or "").strip()
        return s or None

    return surf(sub), surf(obj)


# Relations whose accept-on-error default is flipped to REJECT: causes_or_explains ranges over every
# condition x condition permutation, so a parse glitch that defaulted to accept would leak a junk
# causal pair. Its yield is low and cue-recoverable, so a dropped-on-error causal claim is cheap.
_REJECT_ON_ERROR_RELATIONS = frozenset({"causes_or_explains"})


def build_medgemma_judge(client, *, accept_on_error: bool = True) -> Callable[..., bool]:
    """Wrap an LLMClient-like `.generate(prompt, system=...)` into an accept-biased relation judge.

    accept_on_error=True (recall-first): an unparseable / empty reply accepts rather than dropping a
    possibly-true relation on a formatting glitch -- EXCEPT for _REJECT_ON_ERROR_RELATIONS, where an
    unparseable reply rejects (the combinatorial relations cannot afford accept-on-glitch)."""
    def judge(*, premise: str, claim: str, relation: str | None = None) -> bool:
        on_error = accept_on_error and relation not in _REJECT_ON_ERROR_RELATIONS
        try:
            raw = client.generate(_judge_user_msg(premise, claim), system=_judge_system(relation))
        except Exception:
            return on_error
        match = re.search(r"\{.*\}", raw or "", re.S)
        if not match:
            return on_error
        try:
            return bool(json.loads(match.group(0)).get("asserted"))
        except (ValueError, TypeError):
            return on_error

    return judge


_CONTEXT_JUDGE_SYSTEM = (
    "You are a careful clinical NLP annotator. You are shown one sentence from a clinical document "
    "with the item under test masked as [target item] (other sensitive items may be masked as "
    "[protected item]). Decide ONLY whether the sentence's remaining words establish what clinical "
    "role the masked item plays for this patient — e.g. a condition they have or are assessed for, "
    "a medication they take or are prescribed, a test or procedure performed or ordered, or a "
    "location tied to them.\n"
    "Rules:\n"
    "1. Judge from the surrounding words only; the mask itself tells you nothing.\n"
    "2. A bare mention without a role (a heading, a list fragment, small talk) is NOT informative.\n"
    "3. Do not guess from outside medical knowledge what the masked item might be.\n"
    "Answer strictly as JSON: {\"informative\": true|false, \"why\": \"<=12 words\"}.\n\n"
    "Examples:\n"
    "SENTENCE: \"she was diagnosed with [target item] two years ago and manages it with diet.\"\n"
    "CATEGORY: health condition\n"
    "{\"informative\": true, \"why\": \"diagnosed-with frames it as her condition\"}\n"
    "SENTENCE: \"[target item] , [protected item] .\"\n"
    "CATEGORY: drug\n"
    "{\"informative\": false, \"why\": \"bare list, no role stated\"}"
)


def build_informative_context_judge(client, *, accept_on_error: bool = True) -> Callable[..., bool]:
    """Cue-miss escalation for the semantic_property context-probe gate: does this (already
    redacted) locator sentence establish the masked item's clinical role, so a category probe on
    it is meaningful? Accept-biased on infrastructure/parse failure — the reader three-point gate
    is the real acceptance check, so a spurious accept costs one reader probe, never a bad keep."""
    def judge(*, locator: str, category: str) -> bool:
        prompt = (
            f"SENTENCE:\n{locator}\n\nCATEGORY: {category}\n\n"
            f"Does the sentence establish that [target item] plays a real clinical role as a "
            f"{category} for this patient? Answer JSON only."
        )
        try:
            raw = client.generate(prompt, system=_CONTEXT_JUDGE_SYSTEM)
        except Exception:
            return accept_on_error
        match = re.search(r"\{.*\}", raw or "", re.S)
        if not match:
            return accept_on_error
        try:
            return bool(json.loads(match.group(0)).get("informative"))
        except (ValueError, TypeError):
            return accept_on_error

    return judge


class RelationSupportCascade:
    """Accept-only escalator for cue-miss opportunities. See module docstring."""

    def __init__(
        self,
        judge: Callable[..., bool],
        *,
        mednli_entail: Callable[[str, str], float] | None = None,
        mednli_threshold: float = 0.98,
        max_workers: int = 6,
    ):
        self._judge = judge
        self._mednli = mednli_entail
        self._mednli_threshold = float(mednli_threshold)
        self._max_workers = int(max_workers)

    def judge_batch(self, calls: Sequence[Mapping]) -> list[bool]:
        """Concurrent MedGemma-only verdicts (fills the model's request slots). The miner calls this
        for cue-miss batches. MedGemma-only by design: the accept-only MedNLI tier over-entails and,
        as measured, floods the yield with junk (~5% of its accepts were real), so it is deliberately
        NOT used here — the accurate judge decides every candidate, made affordable by concurrency."""
        from concurrent.futures import ThreadPoolExecutor

        def decide(call: Mapping) -> bool:
            subject, object_ = _subject_object_surfaces(call.get("arguments") or [])
            if not subject or not object_:
                return False
            relation = str(call.get("relation"))
            claim = _claim(relation, subject, object_)
            if claim is None:
                return False
            return bool(self._judge(
                premise=str(call.get("quote", "")), claim=claim, relation=relation))

        if not calls:
            return []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            return list(pool.map(decide, calls))

    def __call__(
        self, *, relation: str, quote: str, arguments: Sequence[Mapping],
        anchor_kind: str | None = None, document: str | None = None,
    ) -> bool:
        subject, object_ = _subject_object_surfaces(arguments)
        if not subject or not object_:
            return False  # no judgeable pair -> defer to cue-only (no recovery, no regression)
        claim = _claim(relation, subject, object_)
        if claim is None:
            return False
        premise = quote  # ponytail: anchor's surgical stitch; widen from `document` if recall needs it
        if self._mednli is not None:
            # NLI hypothesis mirrors the claim's directionality; object-first relations read the same.
            hypothesis = f"{claim}."
            try:
                if self._mednli(premise, hypothesis) >= self._mednli_threshold:
                    return True  # cheap high-confidence recovery; accept-only, never rejects
            except Exception:
                pass  # best-effort; the judge still decides (MedNLI can never cause a false-negative)
        return bool(self._judge(premise=premise, claim=claim, relation=relation))
