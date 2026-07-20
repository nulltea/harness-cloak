"""Prefilter-vs-current regression test for relation-support opportunities (throwaway spike).

Tests the proposed LLM set-call PREFILTER against the current gazetteer proposer, holding the rest of
the pipeline (grounding + lexical cue + MedGemma judge) FIXED -- only the context-candidate source
differs. So any delta is purely from candidate recall, and A\\B is a true regression (a pair the
current full pipeline accepts that the prefilter drops).

  A (current)   = relation_support_opportunities(..., escalator)      -- gazetteer context candidates
  B (prefilter) = same, with relation_context_candidates monkeypatched to the set-call phrases

Per (controlled condition, relation) we run one enumeration set-call on doc_orig, type each returned
phrase by the relation it came from, locate it verbatim in the source, and feed it as a context
candidate. Reader + judge are the local MedGemma (free). One GPU process.

Run: CLOAK_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
        scripts/spikes/relation_prefilter_regression.py
"""
from __future__ import annotations

import collections
import json
import re

import cloak.train.qa_builder as qb
from cloak.corpora import load_task_docs
from cloak.llm import LLMClient
from cloak.train.qa_builder import canon, relation_support_opportunities, relation_teacher_span_inventory
from cloak.train.reward import QA_BASE_URL

import build_qa_utility_artifact as bqa  # _build_relation_support_escalator, RELATION_SUPPORT_JUDGE_MODEL

ENV = "results/qa_v2_aci_full/ranker-env.json"
N_DOCS = 5
# per-relation set-call: {kind}, {phrase}, and the runtime_type to stamp on returned phrases so they
# type into the right relation class (test->monitoring, drug->treatment, symptom->symptom, ...).
RELATION_SETCALL = {
    "prescribed_with": ("medication or drug", "was prescribed, started, continued, or given to treat", "drug"),
    "procedure_for": ("medical procedure, therapy, surgery, or referral",
                      "was performed, planned, referred, or previously done to treat", "procedure"),
    "tests_for": ("diagnostic test, lab, panel, imaging study, or exam",
                  "was ordered, performed, or resulted to work up, monitor, or evaluate", "test"),
    "contraindicated_because_of": ("medication, drug, drug class, or procedure",
                                   "must be avoided or is contraindicated because of", "drug"),
    "causes_or_explains": ("condition, symptom, or finding", "is caused or explained by", "symptom"),
}
FRAME = ('List EVERY distinct {kind} that the note says {phrase} the patient\'s {anchor}.\n'
         'Copy each answer verbatim as a short phrase from the note; include nothing not in the note.\n'
         'Respond with ONLY a JSON array of strings, e.g. ["x","y"]. If there are none, respond [].')


def parse_array(raw: str):
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [str(x).strip() for x in arr] if isinstance(arr, list) else []


def fk_key(fk) -> str:
    return json.dumps(fk, sort_keys=True, default=str)


def main():
    env = json.load(open(ENV))["frozen_environment"]["documents"]
    doc_ids = list(env)[:N_DOCS]
    notes = {r["id"]: r["text"] for r in load_task_docs("clinical")}
    escalator = bqa._build_relation_support_escalator()
    reader = LLMClient(bqa.RELATION_SUPPORT_JUDGE_MODEL, base_url=QA_BASE_URL, api_key="x",
                       temperature=0.0, max_tokens=128,
                       extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    tot = collections.Counter()
    dropped_detail, gained_detail = [], []
    orig_ctx_fn = qb.relation_context_candidates

    for doc_id in doc_ids:
        env_doc = env[doc_id]
        src = notes[doc_id]
        # A: current gazetteer pipeline
        A = {fk_key(o["fact_key"]): o for o in relation_support_opportunities(src, env_doc, escalator=escalator)}

        # prefilter: 1 set-call per (controlled condition, relation) on doc_orig
        conditions = {}
        for occ in env_doc.get("occurrences", []):
            if occ.get("controlled") and occ.get("runtime_type") == "health-condition" and occ.get("surface"):
                conditions.setdefault(str(occ.get("decision_id")), str(occ["surface"]))
        cand = {}  # (rtype,start,end,literal) -> context candidate
        for _dec, anchor in conditions.items():
            for rel, (kind, phrase, rtype) in RELATION_SETCALL.items():
                prompt = "Clinical note:\n\"\"\"\n" + src + "\n\"\"\"\n\n" + FRAME.format(
                    kind=kind, phrase=phrase, anchor=anchor)
                for item in parse_array(reader.generate(prompt)):
                    m = re.search(re.escape(item), src, re.IGNORECASE)
                    if not m:
                        continue  # not verbatim-locatable -> cannot ground
                    s, e = m.span()
                    key = (rtype, s, e, src[s:e])
                    cand.setdefault(key, {
                        "context_candidate_id": "context:" + qb._stable_hash(
                            {"runtime_type": rtype, "start": s, "end": e, "literal": src[s:e]}),
                        "kind": "context_literal", "runtime_type": rtype,
                        "literal": src[s:e], "start": s, "end": e, "provenance": "llm_prefilter"})
        prefilter_candidates = sorted(cand.values(), key=lambda r: (r["start"], r["context_candidate_id"]))

        # B: same pipeline, context candidates = prefilter phrases UNION the gazetteer (augment mode).
        # Union guarantees the candidate set is a superset of the current one, so no accepted pair can
        # be dropped -- the regression can only come from REPLACE mode (set _c=list(_c) alone).
        qb.relation_context_candidates = (
            lambda _doc, _c=prefilter_candidates: list(_c) + orig_ctx_fn(_doc))
        try:
            B = {fk_key(o["fact_key"]): o for o in relation_support_opportunities(src, env_doc, escalator=escalator)}
        finally:
            qb.relation_context_candidates = orig_ctx_fn

        inter = set(A) & set(B)
        dropped = set(A) - set(B)
        gained = set(B) - set(A)
        tot["A"] += len(A); tot["B"] += len(B); tot["inter"] += len(inter)
        tot["dropped"] += len(dropped); tot["gained"] += len(gained)
        for k in dropped:
            o = A[k]
            dropped_detail.append((doc_id.split("/")[-1], o["relation"], o.get("recovered_by_escalation"), o["fact_key"]))
        for k in gained:
            o = B[k]
            gained_detail.append((doc_id.split("/")[-1], o["relation"], o.get("recovered_by_escalation"), o["fact_key"]))
        print(f"{doc_id.split('/')[-1]}: A={len(A)} B={len(B)} shared={len(inter)} "
              f"DROPPED={len(dropped)} gained={len(gained)} | prefilter_literals={len(prefilter_candidates)}")

    print(f"\n=== TOTAL over {N_DOCS} docs ===")
    print(f"current accepted (A): {tot['A']}  | prefilter accepted (B): {tot['B']}  | shared: {tot['inter']}")
    print(f"DROPPED by prefilter (regression, want 0): {tot['dropped']}")
    print(f"GAINED by prefilter (new recoveries): {tot['gained']}")
    print("\nDROPPED pairs (regressions):")
    for d in dropped_detail:
        print("   ", d[0], d[1], "judge" if d[2] else "cue", d[3])
    print("\nsample GAINED pairs:")
    for d in gained_detail[:20]:
        print("   ", d[0], d[1], "judge" if d[2] else "cue", d[3])


if __name__ == "__main__":
    main()
