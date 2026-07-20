"""Feasibility probe for Codex's set-valued QA mechanism (throwaway spike).

The whole mechanism hinges on ONE unverified assumption (Codex's own caveat): can the
local reader (medgemma-4b-it) reliably ENUMERATE all K answers of a multi-answer relation
as a JSON array? If it can't even do this on the ORIGINAL source note (full information,
no perturbation -- the upper bound), the mechanism is dead on arrival and there is no point
building the grouping/compile/gate/scorer components + pin bumps.

This probe: for every ambiguous (relation, subject) group with >=2 distinct grounded objects
in results/qa_v2_aci_full_v14d, ask the reader an exhaustive "list every X" question against
the raw source note and measure JSON-formedness + member recall/precision. Reader-only, local,
free (no teacher, no paid). One GPU process.

Run: CLOAK_LLM_CACHE=data/llm_cache PYTHONPATH=src .venv/bin/python -u \
        scripts/spikes/set_valued_enumeration_probe.py
"""
from __future__ import annotations

import collections
import json
import re

from cloak.corpora import load_task_docs
from cloak.llm import LLMClient
from cloak.train.qa_builder import canon
from cloak.train.reward import QA_BASE_URL, QA_MODEL

ARTIFACT = "results/qa_v2_aci_full_v14d/aci_full.utility.json"
# gate reasons that reached the reader (= grounded), so the object is a real doc-supported answer
GROUNDED = {"three_point_gate_failed", "unstable", "placeholder_answerable",
            "answer_leakage", "duplicate_fact_group"}
KIND = {"tests_for": "diagnostic test or lab", "prescribed_with": "medication or drug",
        "procedure_for": "procedure or therapy"}
PHRASE = {"tests_for": "ordered to test for, work up, or evaluate",
          "prescribed_with": "prescribed or given to treat",
          "procedure_for": "performed or planned to treat"}

# Authored here (not delegated): exhaustive wrapper over a singular relation question. States
# no K, enumerates no answers; demands note-copied spans as a bare JSON array.
PROMPT = """Clinical note:
\"\"\"
{ctx}
\"\"\"

List EVERY distinct {kind} that the note says was {phrase} the patient's {subject}.
Copy each answer verbatim as a short phrase from the note; include nothing not in the note.
Respond with ONLY a JSON array of strings, e.g. ["x","y"]. If there are none, respond [].
Answer:"""


def toks(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", canon(s)) if len(t) > 1}


def matches(pred: str, gold: str) -> bool:
    a, b = toks(pred), toks(gold)
    if not a or not b:
        return False
    if a <= b or b <= a:
        return True
    return len(a & b) / len(a | b) >= 0.5


def parse_array(raw: str) -> list[str] | None:
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return [str(x) for x in arr] if isinstance(arr, list) else None


def main() -> None:
    art = json.load(open(ARTIFACT))
    docs = art["documents"]
    rg = art["relation_generation"]
    notes = {r["id"]: r["text"] for r in load_task_docs("clinical")}

    def occ_map(doc_id):
        return {str(o["occurrence_id"]): o for o in (docs.get(doc_id) or {}).get("occurrences") or []}

    # gold surface (note text) per object: raw surface for linked, literal for context
    def obj_surface(occ, a):
        if a.get("kind") == "context":
            return str(a.get("literal") or "")
        o = occ.get(str(a.get("occurrence_id"))) or {}
        return str(o.get("surface") or a.get("support_property") or "")

    def obj_id(occ, a):
        if a.get("kind") == "context":
            return ("lit", canon(str(a.get("literal") or "")))
        o = occ.get(str(a.get("occurrence_id"))) or {}
        return ("dec", str(o.get("decision_id")))

    def subj_surface(occ, a):
        if a.get("kind") != "context":
            o = occ.get(str(a.get("occurrence_id"))) or {}
            if o.get("surface"):
                return str(o["surface"])
        return str(a.get("support_property") or a.get("literal") or "")

    # build groups: (doc, relation, subject_decision) -> {obj_id: surface}, + whether it has a residual loss
    groups: dict = {}
    residual = collections.defaultdict(bool)
    for doc_id, recs in rg.items():
        occ = occ_map(doc_id)
        for r in recs:
            a = r.get("arguments") or []
            if len(a) != 2:
                continue
            rel = r.get("relation")
            if rel not in KIND:
                continue
            role = str(r.get("answer_role") or "object")
            subj = a[1] if role == "subject" else a[0]
            obj = a[0] if role == "subject" else a[1]
            if subj.get("kind") != "linked":
                continue  # subject must be a locatable condition
            status = r.get("status")
            if not (status == "kept" or r.get("reason") in GROUNDED):
                continue
            key = (doc_id, rel, str((occ.get(str(subj.get("occurrence_id"))) or {}).get("decision_id")))
            g = groups.setdefault(key, {"subj": subj_surface(occ, subj), "members": {}})
            g["members"][obj_id(occ, obj)] = obj_surface(occ, obj)
            if status != "kept":
                residual[key] = True

    multi = {k: g for k, g in groups.items() if len(g["members"]) >= 2}
    print(f"ambiguous groups (>=2 grounded members): {len(multi)} "
          f"| with a residual loss: {sum(residual[k] for k in multi)}")

    client = LLMClient(QA_MODEL, base_url=QA_BASE_URL, api_key="x", temperature=0.0,
                       max_tokens=96,
                       extra_body={"chat_template_kwargs": {"enable_thinking": False},
                                   "cache_prompt": True})

    json_ok = recalls = precisions = full = n = 0
    rows = []
    for (doc_id, rel, _), g in sorted(multi.items()):
        members = [s for s in g["members"].values() if s.strip()]
        if len(members) < 2:
            continue
        n += 1
        q = PROMPT.format(ctx=notes[doc_id], kind=KIND[rel], phrase=PHRASE[rel], subject=g["subj"])
        raw = client.generate(q)
        arr = parse_array(raw)
        if arr is None:
            rows.append((doc_id, rel, g["subj"], "BAD_JSON", raw[:80]))
            continue
        json_ok += 1
        rec = sum(any(matches(p, m) for p in arr) for m in members) / len(members)
        prec = (sum(any(matches(p, m) for m in members) for p in arr) / len(arr)) if arr else 0.0
        recalls += rec
        precisions += prec
        full += rec == 1.0
        rows.append((doc_id, rel, g["subj"], f"R={rec:.2f} P={prec:.2f}",
                     f"gold={members} pred={arr}"))

    print(f"\ngroups probed: {n}")
    print(f"JSON-formed: {json_ok}/{n} ({json_ok/n:.0%})" if n else "no groups")
    if json_ok:
        print(f"mean member recall (JSON-formed): {recalls/json_ok:.2f}")
        print(f"mean precision  (JSON-formed): {precisions/json_ok:.2f}")
        print(f"fully-enumerated groups: {full}/{json_ok} ({full/json_ok:.0%})")
    print("\nper-group:")
    for r in rows:
        print("  ", r[0].split("/")[-1], r[1], f"[{r[2]}]", r[3], "|", r[4])


if __name__ == "__main__":
    main()
