"""Faithful feasibility probe for the set-valued QA mechanism (throwaway spike).

Mirrors the real reader gate instead of a hand-rolled matcher:
  * orig render  = doc_orig (raw note)
  * rep render   = doc_p, every controlled span -> its NEAREST legal level (greedy-injective:
                   a colliding fill bumps to the next coarser distinct level, else keep)
  * placeholder  = every controlled span -> placeholder
  * scoring      = the gate's own `_context_answer_score` (chain-alias resolution + property
                   entailment for linked members; `_answer_score`=max fact_score for literals),
                   one prediction matches at most one member (one-to-one).
For each ambiguous (relation, subject) group (>=2 distinct grounded objects), ask the exhaustive
set question and measure per-member recall on orig AND rep, placeholder leakage, and the multi-gate
verdict (pass iff every member scores >=t on orig AND on rep AND <t on placeholder).

Reader-only, local, free. One GPU process.
Run: CLOAK_LLM_CACHE=data/llm_cache PYTHONPATH=src .venv/bin/python -u \
        scripts/spikes/set_valued_gate_probe.py
"""
from __future__ import annotations

import collections
import json
import re

from cloak.corpora import load_task_docs
from cloak.llm import LLMClient
from cloak.train.qa_builder import (_context_answer_score, _ordered_decision_levels,
                                    canon, render_frozen_action_vector)
from cloak.train.reward import QA_BASE_URL, QA_MODEL

ARTIFACT = "results/qa_v2_aci_full_v14d/aci_full.utility.json"
THRESHOLD = 1.0
GROUNDED = {"three_point_gate_failed", "unstable", "placeholder_answerable",
            "answer_leakage", "duplicate_fact_group"}
KIND = {"tests_for": "diagnostic test or lab", "prescribed_with": "medication or drug",
        "procedure_for": "procedure or therapy"}
PHRASE = {"tests_for": "ordered to test for, work up, or evaluate",
          "prescribed_with": "prescribed or given to treat",
          "procedure_for": "performed or planned to treat"}
# Authored here: exhaustive wrapper referencing the subject by its sanitized LEVEL (never raw surface).
PROMPT = """Clinical note:
\"\"\"
{ctx}
\"\"\"

List EVERY distinct {kind} that the note says was {phrase} the patient's {subject}.
Copy each answer verbatim as a short phrase from the note; include nothing not in the note.
Respond with ONLY a JSON array of strings, e.g. ["x","y"]. If there are none, respond [].
Answer:"""


def parse_array(raw: str) -> list[str] | None:
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return [str(x) for x in arr] if isinstance(arr, list) else None


def action_id(dec: dict, mode: str, fill: str | None = None) -> str | None:
    for a in dec.get("actions", []):
        if not a.get("legal", True) or a.get("mode") != mode:
            continue
        if fill is None or canon(str(a.get("fill", ""))) == canon(fill):
            return str(a["action_id"])
    return None


def build_vectors(doc: dict) -> tuple[dict, dict, dict[str, str]]:
    """Return (rep_vector, placeholder_vector, rep_level_by_decision). Greedy-injective rep."""
    rep, ph, rep_level = {}, {}, {}
    used_fills: set[str] = set()
    for dec in doc.get("decisions", []):
        did = str(dec["decision_id"])
        keep = action_id(dec, "keep")
        placeholder = action_id(dec, "placeholder")
        # placeholder vector: hide controlled spans, keep the rest
        ph[did] = placeholder if (dec.get("controlled") and placeholder) else (keep or placeholder)
        # rep vector: finest legal level whose fill is not already used; else coarser; else keep/ph
        assigned = None
        if dec.get("controlled"):
            for level in _ordered_decision_levels(dec):
                if canon(level) in used_fills:
                    continue
                aid = action_id(dec, "level", level)
                if aid is not None:
                    assigned, rep[did], rep_level[did] = level, aid, level
                    used_fills.add(canon(level))
                    break
        if assigned is None:
            rep[did] = keep or placeholder
    return rep, ph, rep_level


def one_to_one_recall(preds: list[str], member_rows: list[dict], chains: dict) -> float:
    if not member_rows:
        return 0.0
    used = [False] * len(preds)
    hit = 0
    for row in member_rows:
        for i, p in enumerate(preds):
            if used[i]:
                continue
            if _context_answer_score(row, p, chains) >= THRESHOLD:
                used[i] = True
                hit += 1
                break
    return hit / len(member_rows)


def main() -> None:
    art = json.load(open(ARTIFACT))
    docs = art["documents"]
    rg = art["relation_generation"]
    notes = {r["id"]: r["text"] for r in load_task_docs("clinical")}

    def occ_map(doc_id):
        return {str(o["occurrence_id"]): o for o in docs[doc_id]["occurrences"]}

    # build groups: (doc, rel, subj_dec) -> {subj_level, members{id:(kind,dec_or_lit,support)}}
    groups: dict = {}
    for doc_id, recs in rg.items():
        occ = occ_map(doc_id)
        for r in recs:
            a = r.get("arguments") or []
            if len(a) != 2 or r.get("relation") not in KIND:
                continue
            role = str(r.get("answer_role") or "object")
            subj = a[1] if role == "subject" else a[0]
            obj = a[0] if role == "subject" else a[1]
            if subj.get("kind") != "linked":
                continue
            if not (r.get("status") == "kept" or r.get("reason") in GROUNDED):
                continue
            so = occ.get(str(subj.get("occurrence_id"))) or {}
            key = (doc_id, r["relation"], str(so.get("decision_id")))
            g = groups.setdefault(key, {"subj": str(subj.get("support_property") or ""),
                                        "members": {}, "literals": set()})
            if obj.get("kind") == "context":
                lit = str(obj.get("literal") or "")
                if lit:
                    g["literals"].add(canon(lit))  # uncontrolled -> pre-excluded from privacy gating
            else:
                oo = occ.get(str(obj.get("occurrence_id"))) or {}
                did = str(oo.get("decision_id"))
                g["members"][("dec", did)] = ("linked", did, str(obj.get("support_property") or ""))

    # A set-valued privacy gate only covers CONTROLLED (linked) members; literals are not private.
    # Groups need >=2 distinct controlled objects to be a genuine multi-answer privacy query.
    multi = {k: g for k, g in groups.items() if len(g["members"]) >= 2}
    with_literals_only = sum(1 for g in groups.values() if not g["members"] and g["literals"])
    with_one_linked = sum(1 for g in groups.values() if len(g["members"]) == 1)

    client = LLMClient(QA_MODEL, base_url=QA_BASE_URL, api_key="x", temperature=0.0, max_tokens=128,
                       extra_body={"chat_template_kwargs": {"enable_thinking": False},
                                   "cache_prompt": True})

    # render caches per doc
    rendered: dict[str, tuple[str, str, str]] = {}

    def renders(doc_id):
        if doc_id not in rendered:
            doc = docs[doc_id]
            rep_v, ph_v, _ = build_vectors(doc)
            try:
                rep = render_frozen_action_vector(notes[doc_id], doc, rep_v)[0]
            except ValueError as e:
                rep = f"[REP RENDER FAILED: {e}]"
            try:
                ph = render_frozen_action_vector(notes[doc_id], doc, ph_v)[0]
            except ValueError as e:
                ph = f"[PH RENDER FAILED: {e}]"
            rendered[doc_id] = (notes[doc_id], rep, ph)
        return rendered[doc_id]

    agg = collections.defaultdict(float)
    n = passes = 0
    kindct = collections.Counter()
    rows = []
    for (doc_id, rel, subj_dec), g in sorted(multi.items()):
        doc = docs[doc_id]
        rep_v, ph_v, rep_level = build_vectors(doc)
        chains = {str(d["decision_id"]): d.get("semantic_chain") or [] for d in doc["decisions"]}
        # member rows for the gate scorer
        member_rows, tags = [], []
        for kind, did, support in g["members"].values():
            kindct[kind] += 1
            if kind == "linked":
                req = rep_level.get(did, support)  # level actually shown in the rep render
                member_rows.append({"answer_target": {"kind": "linked_decision",
                                                       "decision_id": did, "required_property": req}})
                tags.append(f"{support}->{req}")
            else:
                member_rows.append({"answer_target": {"kind": "literal", "expected_values": [support]}})
                tags.append(f"lit:{support}")
        orig_ctx, rep_ctx, ph_ctx = renders(doc_id)
        q = lambda ctx: PROMPT.format(ctx=ctx, kind=KIND[rel], phrase=PHRASE[rel], subject=g["subj"])
        po = parse_array(client.generate(q(orig_ctx)))
        pr = parse_array(client.generate(q(rep_ctx)))
        pp = parse_array(client.generate(q(ph_ctx)))
        if po is None or pr is None or pp is None:
            rows.append((doc_id, rel, g["subj"], "BAD_JSON", ""))
            continue
        n += 1
        r_orig = one_to_one_recall(po, member_rows, chains)
        r_rep = one_to_one_recall(pr, member_rows, chains)
        r_ph = one_to_one_recall(pp, member_rows, chains)
        agg["orig"] += r_orig
        agg["rep"] += r_rep
        agg["ph"] += r_ph
        gate = (r_orig == 1.0 and r_rep == 1.0 and r_ph < 1.0)
        passes += gate
        rows.append((doc_id, rel, g["subj"],
                     f"orig={r_orig:.2f} rep={r_rep:.2f} ph={r_ph:.2f} {'PASS' if gate else 'fail'}",
                     f"members={tags} pred_rep={pr}"))

    print(f"ambiguous groups with >=2 CONTROLLED (linked) members: {len(multi)}  "
          f"[literals pre-excluded from gating]")
    print(f"dropped: all-literal groups={with_literals_only}, single-linked groups={with_one_linked} "
          f"(these need the deferred reverse-flip, not set-valued)")
    print(f"member kinds scored: {dict(kindct)}")
    print(f"groups scored (JSON-formed all 3 renders): {n}")
    if n:
        print(f"mean per-member recall  orig={agg['orig']/n:.2f}  rep={agg['rep']/n:.2f}  "
              f"placeholder-leak={agg['ph']/n:.2f}")
        print(f"multi-gate PASS (all members orig>=t & rep>=t & ph<t): {passes}/{n} ({passes/n:.0%})")
    print("\nper-group:")
    for r in rows:
        print("  ", r[0].split("/")[-1], r[1], f"[{r[2]}]", r[3])
        print("      ", r[4])


if __name__ == "__main__":
    main()
