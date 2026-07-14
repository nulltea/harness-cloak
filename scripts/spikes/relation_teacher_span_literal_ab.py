"""A/B spike: span<->literal relation yield under two teacher-call architectures.

Variant A (sectioned single call): one request whose wire schema requires two
arrays, span_relations (linked+linked only) and context_relations (exactly one
linked + one literal), with matching prompt sub-tasks.

Variant B (dedicated literal call): a request restricted to span<->literal
relations only, as the hypothetical second call of a two-call contract.

Each draw is compiled through the production compiler (deterministic gates
only; no reader) and scored against the five known literal-end facts in
aci/D2N002. Direct client calls: every draw is fresh, nothing touches the
production llm cache.
"""
import argparse
import json
import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from openai import OpenAI

from cloak.train.qa_builder import (
    RELATION_TEACHER_BASE_URL,
    RELATION_TEACHER_MODEL,
    compile_relational_assertions,
    relation_teacher_prompt,
    relation_teacher_response_format,
)

# (relation, literal substring) for the five explicit span<->literal facts.
TARGETS = (
    ("prescribed_with", "synthroid"),
    ("monitored_by", "thyroid lab"),
    ("causes_or_explains", "knee pain"),
    ("contraindicated_because_of", "anti-inflammatory"),
    ("treated_with", "polycystic"),
)


def _argument_shapes(bound_schema: dict, *, kinds: set[str]) -> list[dict]:
    shapes = bound_schema["properties"]["relations"]["items"]["properties"]["arguments"]["anyOf"]
    return [
        shape for shape in shapes
        if {branch["properties"]["kind"]["const"] for branch in shape["prefixItems"]} == kinds
        or (kinds == {"linked", "context"}
            and "context" in {branch["properties"]["kind"]["const"] for branch in shape["prefixItems"]})
    ]


def sectioned_format(environment: dict, source: str) -> dict:
    base = relation_teacher_response_format(environment, source)
    schema = base["json_schema"]["schema"]
    relation_item = schema["properties"]["relations"]["items"]
    span_item = deepcopy(relation_item)
    span_item["properties"]["arguments"] = {
        "anyOf": _argument_shapes(schema, kinds={"linked"})
    }
    context_item = deepcopy(relation_item)
    context_item["properties"]["arguments"] = {
        "anyOf": _argument_shapes(schema, kinds={"linked", "context"})
    }
    schema["properties"] = {
        "span_relations": {"type": "array", "maxItems": 12, "items": span_item},
        "context_relations": {"type": "array", "maxItems": 12, "items": context_item},
        "candidate_accounting": schema["properties"]["candidate_accounting"],
    }
    schema["required"] = ["span_relations", "context_relations", "candidate_accounting"]
    return base


def dedicated_format(environment: dict, source: str) -> dict:
    base = relation_teacher_response_format(environment, source)
    schema = base["json_schema"]["schema"]
    relation_item = schema["properties"]["relations"]["items"]
    relation_item["properties"]["arguments"] = {
        "anyOf": _argument_shapes(schema, kinds={"linked", "context"})
    }
    return base


SECTIONED_RESPONSE = """RESPONSE
Return two relation lists.
span_relations: relations whose subject and object are both displayed S-labels (kind linked, span_label plus exactly one of that label's listed levels copied verbatim as support_property).
context_relations: relations pairing exactly one linked S-label argument with one uncontrolled argument of kind context, whose literal is exact source text that is not any displayed span.
Emit each distinct fact once, in the list its argument kinds require, at the S-label inside the sentence that states the relation. Never quote a displayed span as a context literal.
Example span_relations record: relation prescribed_with; subject linked S1 with one listed S1 level; object linked S2 with one listed S2 level; question "Which medication category was prescribed for the joint condition?"; accepted answer "opioid analgesic".
Example context_relations record: relation prescribed_with; subject linked S3 with one listed S3 level; object context literal "synthroid" quoted from the relation sentence; accepted answer "synthroid".
Return exactly one candidate_accounting row per S-label covering both lists, with a short reason for every row. emitted means a relation record in either list uses the label; duplicate_mention means another S-label of the same value already carries the fact (name that label in the reason); exhausted_no_relation means no explicit supported relation; unsupported means insufficient source role/connection. Reasons must reference labels and levels only and never repeat displayed span text. Return only the structured response."""

DEDICATED_TASK = """TASK
Find explicit, source-grounded relations that pair one displayed S-label with UNCONTROLLED source text: a phrase stated in the source that is not any displayed span (for example a drug, test, therapy, or symptom the source names without a label). Find as many non-duplicate such relations as the cap (12) permits. Abstain rather than inventing a fact. Do not return relations whose both arguments are displayed S-labels."""

DEDICATED_RESPONSE = """RESPONSE
Each relation record contains: relation; a subject argument then an object argument; a question; accepted answers; the fixed scoring contract.
Exactly one argument is kind linked: span_label set to its S-label and support_property set to exactly one of that label's listed levels, copied verbatim. The other argument is kind context with its exact uncontrolled source text as literal. Never quote a displayed span as a context literal.
Emit each distinct fact once, at the S-label inside the sentence that states the relation.
Example record: relation prescribed_with; subject linked S1 with one listed S1 level as support_property; object context literal "synthroid" quoted from the relation sentence; question "Which medication was continued for the endocrine condition?"; accepted answer "synthroid".
Example record: relation causes_or_explains; subject linked S2 with one listed S2 level; object context literal "knee pain"; question "What symptom is attributed to the joint disease?"; accepted answer "knee pain".
Return exactly one candidate_accounting row per S-label, with a short reason for every row. emitted means a relation record uses the label; duplicate_mention means another S-label of the same value already carries the fact; exhausted_no_relation means no explicit supported relation with uncontrolled text; unsupported means insufficient source role/connection. Reasons must reference labels and levels only and never repeat displayed span text. Return only the structured response."""


def sectioned_prompt(base_prompt: str) -> str:
    head, _, tail = base_prompt.partition("RESPONSE\n")
    _, _, source = tail.partition("SOURCE DOCUMENT\n")
    return f"{head}{SECTIONED_RESPONSE}\n\nSOURCE DOCUMENT\n{source}"


def dedicated_prompt(base_prompt: str) -> str:
    _, _, after_task = base_prompt.partition("HOW TO INSPECT THE SOURCE")
    head = "HOW TO INSPECT THE SOURCE" + after_task
    head, _, tail = head.partition("RESPONSE\n")
    _, _, source = tail.partition("SOURCE DOCUMENT\n")
    return f"{DEDICATED_TASK}\n\n{head}{DEDICATED_RESPONSE}\n\nSOURCE DOCUMENT\n{source}"


# Experiment retry budget — see relation_teacher_split_calls.py: the SDK obeys
# OpenRouter's 60s Retry-After on every retry, so a hard per-draw wall-clock
# ceiling (not shorter backoff) is what prevents minutes of stacked waits.
DRAW_BUDGET_S = 90
THROTTLE_WAIT_S = 15


def run_draw(client: OpenAI, prompt: str, response_format: dict) -> dict | None:
    import time
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
            print(f"  attempt {attempt}: request error {type(error).__name__}; {wait:.0f}s to budget", flush=True)
            if wait:
                time.sleep(wait)
            continue
        if resp.choices and (resp.choices[0].message.content or "").strip():
            return json.loads(resp.choices[0].message.content)
        print(f"  attempt {attempt}: empty completion (200 no-choices), immediate retry", flush=True)
    print(f"  gave up after {attempt} attempts / {DRAW_BUDGET_S}s budget", flush=True)
    return None


def score(proposals: list[dict], source: str, environment: dict) -> dict:
    literal_proposals = [
        p for p in proposals
        if any(a.get("kind") == "context" for a in p.get("arguments") or [])
    ]
    accepted, rejected = compile_relational_assertions(
        "aci/D2N002", source, environment, proposals,
    )
    accepted_literals = [
        row for row in accepted
        if any(a.get("kind") == "context" for a in row["evidence"]["arguments"])
    ]

    def hits(rows, get_relation, get_literals):
        found = set()
        for relation, literal_sub in TARGETS:
            for row in rows:
                if get_relation(row) == relation and any(
                    literal_sub in literal.lower() for literal in get_literals(row)
                ):
                    found.add((relation, literal_sub))
        return found

    proposed_hits = hits(
        literal_proposals, lambda p: p.get("relation"),
        lambda p: [str(a.get("literal") or "") for a in p.get("arguments") or []],
    )
    accepted_hits = hits(
        accepted_literals, lambda r: r.get("relation"),
        lambda r: [str(a.get("literal") or "") for a in r["evidence"]["arguments"]],
    )
    return {
        "proposals": len(proposals),
        "literal_proposals": len(literal_proposals),
        "accepted_total": len(accepted),
        "accepted_literal": len(accepted_literals),
        "target_proposed": sorted(f"{r}:{s}" for r, s in proposed_hits),
        "target_accepted": sorted(f"{r}:{s}" for r, s in accepted_hits),
        "reject_reasons": [r["detail_reason"] for r in rejected],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--arms", default="/tmp/task_arms_qa_v2_d2n002.json")
    parser.add_argument("--env", default="/tmp/qa-v2-d2n002-nemotron-r15.json")
    args = parser.parse_args()

    source = json.load(open(args.arms))["clinical"]["aci/D2N002"]["no_privacy"][0]
    environment = json.load(open(args.env))["documents"]["aci/D2N002"]
    base_prompt = relation_teacher_prompt("aci/D2N002", source, environment)
    client = OpenAI(
        base_url=RELATION_TEACHER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        # We own retries via run_draw's wall-clock budget; the SDK's default
        # 600s timeout also reads as a hang, so cap it well under the budget.
        max_retries=0,
        timeout=120,
    )

    variants = {
        "A_sectioned": (sectioned_prompt(base_prompt), sectioned_format(environment, source)),
        "B_dedicated": (dedicated_prompt(base_prompt), dedicated_format(environment, source)),
    }
    for name, (prompt, response_format) in variants.items():
        print(f"\n===== {name} (prompt {len(prompt)} chars) =====", flush=True)
        for draw in range(1, args.draws + 1):
            payload = run_draw(client, prompt, response_format)
            if payload is None:
                print(f"draw {draw}: EMPTY after retries", flush=True)
                continue
            proposals = (
                list(payload.get("span_relations", [])) + list(payload.get("context_relations", []))
                if name == "A_sectioned" else list(payload.get("relations", []))
            )
            result = score(proposals, source, environment)
            print(f"draw {draw}: {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
