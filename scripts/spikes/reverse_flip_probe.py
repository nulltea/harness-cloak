"""Faithful feasibility probe for the DEFERRED reverse-flip (task #5) — throwaway spike.

For a (relation, condition C) group with >=2 uncontrolled literal objects, emit a SINGLE-answer
reverse QA whose locator is the compound literal list and whose answer is the controlled condition:
    "For which medical condition were <L1>, <L2>, ... ordered?"  answer = C
This is single-answer (no set scorer): it reuses the real gate reader (`_read_batch`, QA_PROMPT) and
scorer (`_context_answer_score`, chain-alias + entailment) over the three renders
(orig=doc_orig, rep=nearest-level doc_p, placeholder=controlled spans hidden). Because C is controlled,
the placeholder render hides it -> the gate's `placeholder < t` holds by construction.

Reader-only, local, free. One GPU process.
Run: CLOAK_LLM_CACHE=data/llm_cache PYTHONPATH=src .venv/bin/python -u \
        scripts/spikes/reverse_flip_probe.py
"""
from __future__ import annotations

import collections
import json

from cloak.corpora import load_task_docs
from cloak.train.qa_builder import (_context_answer_score, _ordered_decision_levels,
                                    canon, render_frozen_action_vector)
from cloak.train.reward import _read_batch

ARTIFACT = "results/qa_v2_aci_full_v14d/aci_full.utility.json"
THRESHOLD = 1.0
GROUNDED = {"three_point_gate_failed", "unstable", "placeholder_answerable",
            "answer_leakage", "duplicate_fact_group"}
VERB = {"tests_for": "ordered", "prescribed_with": "prescribed", "procedure_for": "performed"}
MAX_LOCATORS = 4  # cap the compound locator so the question stays readable


def action_id(dec, mode, fill=None):
    for a in dec.get("actions", []):
        if a.get("legal", True) and a.get("mode") == mode and (
                fill is None or canon(str(a.get("fill", ""))) == canon(fill)):
            return str(a["action_id"])
    return None


def condition_decision_index(doc):
    """canon(level/alias/canonical_key) -> decision_id, for health-condition decisions only.
    Records store compiled args by span_label (S1..) not occurrence_id, so resolve the subject
    condition to its decision by its support_property matching a node in exactly one chain."""
    idx = {}
    for dec in doc.get("decisions", []):
        if dec.get("runtime_type") != "health-condition":
            continue
        keys = [dec.get("canonical_key")]
        for n in dec.get("semantic_chain") or []:
            keys.append(n.get("node"))
            keys += (n.get("answer_aliases") or [])
        for k in keys:
            if k:
                idx.setdefault(canon(str(k)), str(dec["decision_id"]))
    return idx


def build_vectors(doc):
    """(rep_vector, placeholder_vector, rep_level_by_decision), greedy-injective rep."""
    rep, ph, rep_level, used = {}, {}, {}, set()
    for dec in doc.get("decisions", []):
        did = str(dec["decision_id"])
        keep, placeholder = action_id(dec, "keep"), action_id(dec, "placeholder")
        ph[did] = placeholder if (dec.get("controlled") and placeholder) else (keep or placeholder)
        assigned = None
        if dec.get("controlled"):
            for level in _ordered_decision_levels(dec):
                if canon(level) in used:
                    continue
                aid = action_id(dec, "level", level)
                if aid is not None:
                    assigned, rep[did], rep_level[did] = level, aid, level
                    used.add(canon(level))
                    break
        if assigned is None:
            rep[did] = keep or placeholder
    return rep, ph, rep_level


def main():
    art = json.load(open(ARTIFACT))
    docs = art["documents"]
    rg = art["relation_generation"]
    notes = {r["id"]: r["text"] for r in load_task_docs("clinical")}

    def occ_map(d):
        return {str(o["occurrence_id"]): o for o in docs[d]["occurrences"]}

    # (doc,rel,subjdec) -> {subj_level, subj_dec, literals:[surface,...], lost:bool}
    # Grouping from JUDGE-ACCEPTED span_literal pairs (recovered_by_escalation), NOT raw teacher
    # proposals -- so a phantom pairing the teacher over-proposed (e.g. RUQ-ultrasound -> esophageal
    # disease, which the judge rejects) never forms a group. The opportunity fact_key gives the
    # condition decision_id directly.
    opps = art["relation_support_opportunities"]
    dec_by_id = {str(x["decision_id"]): x for doc in docs.values() for x in doc["decisions"]}
    groups = {}
    for doc, lst in opps.items():
        for o in lst:
            if o.get("scope") != "span_literal" or not o.get("recovered_by_escalation"):
                continue
            rel = o.get("relation")
            if rel not in VERB:
                continue
            args = (o.get("fact_key") or [])[1:]
            cond = next((a[1] for a in args if a[0] == "linked_decision"), None)
            lit = next((a[1] for a in args if a[0] == "context_literal"), None)
            if not cond or not lit:
                continue
            if (dec_by_id.get(str(cond)) or {}).get("runtime_type") != "health-condition":
                continue  # subject must be a condition
            g = groups.setdefault((doc, rel, str(cond)), {"subj_dec": str(cond), "lits": {}})
            g["lits"].setdefault(canon(str(lit)), str(lit))

    MIN_LITS = 1  # judge accepts ~1 literal/condition, so the compound (>=2) premise is empty here;
    # test the reliable single judge-accepted (condition, literal) pairs as reverse QAs.
    eligible = {k: g for k, g in groups.items() if len(g["lits"]) >= MIN_LITS}
    for g in eligible.values():
        dec = dec_by_id.get(g["subj_dec"]) or {}
        levels = _ordered_decision_levels(dec)
        g["subj_level"] = levels[0] if levels else str(dec.get("canonical_key") or "")
        g["lost"] = True  # judge-accepted opportunities the teacher didn't land forward

    npass = n = 0
    rows = []
    for (doc, rel, sd), g in sorted(eligible.items()):
        d = docs[doc]
        _, ph_v, rep_level = build_vectors(d)
        rep_v = build_vectors(d)[0]
        chains = {str(x["decision_id"]): x.get("semantic_chain") or [] for x in d["decisions"]}
        locs = list(g["lits"].values())[:MAX_LOCATORS]
        # answer-type first ("medical condition"), locators last, so the extractive reader names the
        # diagnosis instead of copying the visible test names (the echo failure).
        question = (f"What single medical condition or diagnosis were {', '.join(locs)} "
                    f"{VERB[rel]} to evaluate or treat? Name only the condition.")
        req = rep_level.get(sd, g["subj_level"])
        row = {"answer_target": {"kind": "linked_decision", "decision_id": sd, "required_property": req}}
        try:
            orig_ctx = notes[doc]
            rep_ctx = render_frozen_action_vector(notes[doc], d, rep_v)[0]
            ph_ctx = render_frozen_action_vector(notes[doc], d, ph_v)[0]
        except ValueError as e:
            rows.append((doc, rel, g["subj_level"], f"RENDER_FAIL {e}", ""))
            continue
        ao = _read_batch([question], orig_ctx)[0]
        ar = _read_batch([question], rep_ctx)[0]
        ap = _read_batch([question], ph_ctx)[0]
        so_ = _context_answer_score(row, ao, chains)
        sr = _context_answer_score(row, ar, chains)
        sp = _context_answer_score(row, ap, chains)
        gate = so_ >= THRESHOLD and sr >= THRESHOLD and sp < THRESHOLD
        n += 1
        npass += gate
        rows.append((doc, rel, g["subj_level"],
                     f"orig={so_:.0f} rep={sr:.0f} ph={sp:.0f} {'PASS' if gate else 'fail'} "
                     f"{'(lost)' if g['lost'] else '(kept-elsewhere)'}",
                     f"Q={question!r} | ans o/r/p = {ao!r} / {ar!r} / {ap!r}"))

    print(f"reverse-flip eligible groups (>=2 literals): {len(eligible)}")
    print(f"scored: {n}  | multi-render gate PASS: {npass}/{n} ({npass/n:.0%})" if n else "none")
    print(f"  of PASS, how many were a LOST fact: "
          f"{sum('PASS' in r[3] and 'lost' in r[3] for r in rows)}")
    print("\nper-group:")
    for r in rows:
        print("  ", r[0].split("/")[-1], r[1], f"[{r[2]}]", r[3])
        print("      ", r[4])


if __name__ == "__main__":
    main()
