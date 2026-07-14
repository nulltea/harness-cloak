"""Variant B: two dedicated teacher calls (span<->span, then span<->literal).

Single draw each = 2 live calls total. Prints span-pair and literal distinct
facts from each call and the combined compile outcome, so the two-call split
can be compared against the sectioned single call on identical D2N002 inputs.
Direct client calls; nothing touches the production llm cache.
"""
import json
import os
import sys
import time
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from openai import OpenAI

from cloak.train.qa_builder import (
    RELATION_TEACHER_BASE_URL,
    RELATION_TEACHER_MODEL,
    compile_relational_assertions,
    relation_teacher_prompt,
    relation_teacher_response_format,
    relation_teacher_span_inventory,
)

TARGETS = (
    ("prescribed_with", "ultram"),      # the only span<->span target
    ("prescribed_with", "synthroid"),
    ("monitored_by", "thyroid lab"),
    ("causes_or_explains", "knee pain"),
    ("contraindicated_because_of", "anti-inflammatory"),
    ("treated_with", "polycystic"),
)


def _single_list_format(environment, source, kind_pairs):
    """A response_format with one `relations` array bound to the given pair shapes."""
    sectioned = relation_teacher_response_format(environment, source)
    schema = sectioned["json_schema"]["schema"]
    span_item = schema["properties"]["span_relations"]["items"]
    context_item = schema["properties"]["context_relations"]["items"]
    template = deepcopy(span_item)
    span_shapes = span_item["properties"]["arguments"]["anyOf"]
    context_shapes = context_item["properties"]["arguments"]["anyOf"]
    by_pair = {}
    for shape in span_shapes + context_shapes:
        key = tuple(b["properties"]["kind"]["const"] for b in shape["prefixItems"])
        by_pair[key] = shape
    template["properties"]["arguments"] = {"anyOf": [by_pair[p] for p in kind_pairs]}
    schema["properties"] = {
        "relations": {"type": "array", "maxItems": 12, "items": template},
        "candidate_accounting": schema["properties"]["candidate_accounting"],
    }
    schema["required"] = ["relations", "candidate_accounting"]
    return sectioned


def _swap_response(base_prompt, task_line, response_block):
    head, _, tail = base_prompt.partition("HOW TO INSPECT THE SOURCE")
    body = "HOW TO INSPECT THE SOURCE" + tail
    body, _, rest = body.partition("RESPONSE\n")
    _, _, source = rest.partition("SOURCE DOCUMENT\n")
    return f"{task_line}\n\n{body}RESPONSE\n{response_block}\n\nSOURCE DOCUMENT\n{source}"


SPAN_TASK = """TASK
Find explicit, source-grounded relations whose subject AND object are both displayed S-labels. Do not use uncontrolled literals. Find as many non-duplicate such relations as the cap (12) permits; abstain rather than inventing a fact."""

SPAN_RESPONSE = """Each relation record: relation; subject then object, both kind linked with span_label and one listed level copied verbatim as support_property; a question; accepted answers; the scoring contract. Emit each distinct fact once at the S-label inside the relation's sentence. Questions/answers use the listed levels, never source words. Then one candidate_accounting row per S-label (emitted / duplicate_mention / exhausted_no_relation / unsupported) with a short label-referential reason."""

LITERAL_TASK = """TASK
Find explicit, source-grounded relations that pair exactly one displayed S-label with UNCONTROLLED source text (a phrase the source states that is not any displayed span). Do not return relations whose both arguments are displayed S-labels. Find as many non-duplicate such relations as the cap (12) permits; abstain rather than inventing a fact."""

LITERAL_RESPONSE = """Each relation record: relation; subject then object; a question; accepted answers; the scoring contract. Exactly one argument is kind linked (span_label + one listed level verbatim as support_property); the other is kind context with its exact uncontrolled source text as literal. Never quote a displayed span as a literal. Emit each distinct fact once at the S-label inside the relation's sentence. Then one candidate_accounting row per S-label (emitted / duplicate_mention / exhausted_no_relation / unsupported) with a short label-referential reason."""


# Experiment retry budget. The SDK honors OpenRouter's Retry-After (up to 60s)
# on every retry; stacking those with no ceiling is what wasted minutes. We own
# retries (SDK max_retries=0) and cap the total per-draw wall time so a
# throttled draw fails fast instead of sleeping through many 60s waits. A failed
# draw is acceptable for a spike; production llm.py keeps its patient default.
DRAW_BUDGET_S = 90
THROTTLE_WAIT_S = 15


def draw(client, prompt, response_format):
    deadline = time.monotonic() + DRAW_BUDGET_S
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            resp = client.chat.completions.create(
                model=RELATION_TEACHER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format=response_format,
                extra_body={"reasoning": {"exclude": True}},
            )
        except Exception as error:
            wait = min(THROTTLE_WAIT_S, max(0, deadline - time.monotonic()))
            print(f"  attempt {attempt}: {type(error).__name__}; {wait:.0f}s to budget", flush=True)
            if wait:
                time.sleep(wait)
            continue
        if resp.choices and (resp.choices[0].message.content or "").strip():
            return json.loads(resp.choices[0].message.content).get("relations", [])
        print(f"  attempt {attempt}: empty completion (200 no-choices), immediate retry", flush=True)
    print(f"  gave up after {attempt} attempts / {DRAW_BUDGET_S}s budget", flush=True)
    return None


def distinct_facts(relations, label_to_surface):
    span, literal = set(), set()
    for p in relations:
        args = p.get("arguments") or []
        kinds = [a.get("kind") for a in args]
        if kinds == ["linked", "linked"]:
            span.add((p["relation"], tuple(sorted(
                label_to_surface.get(a["span_label"], a["span_label"]) for a in args))))
        elif "context" in kinds and "linked" in kinds:
            literal.add((p["relation"], tuple(sorted(
                str(a.get("literal") or label_to_surface.get(a.get("span_label"), "")) for a in args))))
    return span, literal


def main():
    source = json.load(open("/tmp/task_arms_qa_v2_d2n002.json"))["clinical"]["aci/D2N002"]["no_privacy"][0]
    environment = json.load(open("/tmp/qa-v2-d2n002-nemotron-r15.json"))["documents"]["aci/D2N002"]
    base = relation_teacher_prompt("aci/D2N002", source, environment)
    label_to_surface = {r["span_label"]: r["surface"] for r in relation_teacher_span_inventory(environment)}
    client = OpenAI(base_url=RELATION_TEACHER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"],
                    max_retries=0, timeout=120)

    print("=== CALL 1: span<->span ===", flush=True)
    span_rel = draw(client, _swap_response(base, SPAN_TASK, SPAN_RESPONSE),
                    _single_list_format(environment, source, [("linked", "linked")]))
    print("=== CALL 2: span<->literal ===", flush=True)
    lit_rel = draw(client, _swap_response(base, LITERAL_TASK, LITERAL_RESPONSE),
                   _single_list_format(environment, source, [("linked", "context"), ("context", "linked")]))

    span_rel, lit_rel = span_rel or [], lit_rel or []
    s1, _ = distinct_facts(span_rel, label_to_surface)
    _, l2 = distinct_facts(lit_rel, label_to_surface)
    combined = list(span_rel) + list(lit_rel)
    accepted, rejected = compile_relational_assertions("aci/D2N002", source, environment, combined)
    acc_span = [r for r in accepted if all(a["kind"] == "linked" for a in r["evidence"]["arguments"])]
    acc_lit = [r for r in accepted if any(a["kind"] == "context" for a in r["evidence"]["arguments"])]
    hit = lambda rows: sorted({f"{r}:{s}" for r, s in TARGETS
                               for row in rows
                               if row["relation"] == r
                               and any(s in str(a.get("literal") or label_to_surface.get(a.get("span_label"), "")).lower()
                                       for a in row["evidence"]["arguments"])})

    print(json.dumps({
        "call1_span_proposals": len(span_rel),
        "call1_distinct_span_facts": len(s1),
        "call2_literal_proposals": len(lit_rel),
        "call2_distinct_literal_facts": len(l2),
        "accepted_span": len(acc_span),
        "accepted_literal": len(acc_lit),
        "targets_accepted": hit(accepted),
        "reject_reasons": [r["detail_reason"] for r in rejected],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
