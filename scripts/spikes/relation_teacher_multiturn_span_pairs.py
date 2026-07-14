"""Prompt spike: multi-turn span<->span relations within one problem block.

D2N002 holds a second true span-pair fact, monitored_by(arthritis ->
autoimmune panel), whose arguments sit in different speaker turns of the SAME
problem discussion (a patient "okay." between them). The production anchor
rejects cross-turn links, so this spike measures, per draw:

1. proposal: does the teacher emit the multi-turn span pair when the prompt
   permits problem-block scope (never across problem switches)?
2. spike grounding: a problem-block anchor (both spans inside one
   assessment/plan block for a single problem, relation cue present) --
   deterministic, spike-local, not production code.
3. production compile: expected to reject the multi-turn pair (contrast).

Budget: --draws live calls, 90s wall-clock each, direct client (no cache).
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from openai import OpenAI

from cloak.train.qa_builder import (
    ACI_RELATION_CONTRACT,
    RELATION_TEACHER_BASE_URL,
    RELATION_TEACHER_MODEL,
    compile_relational_assertions,
    relation_teacher_prompt,
    relation_teacher_response_format,
    relation_teacher_span_inventory,
)

DRAW_BUDGET_S = 90
THROTTLE_WAIT_S = 15

MULTITURN_RULE = """A relation may connect spans from different turns of the SAME problem discussion: the block where the doctor assesses and plans one problem, including short patient acknowledgments between the doctor's sentences. Never link spans from different problem discussions or from unrelated small talk.
Example (generic): "... this looks like [S1: condition] ... [patient] okay . [doctor] i will also order [S2: test]" => monitored_by(S1, S2) when the test is ordered to evaluate that condition."""

# Spike-local problem-block boundaries for ACI-style dialogue.
_BLOCK_BOUNDARY = re.compile(
    r"my assessment and my plan|for your (?:second|third|next|last) problem",
    re.IGNORECASE,
)


def problem_blocks(source: str) -> list[tuple[int, int]]:
    cuts = [m.start() for m in _BLOCK_BOUNDARY.finditer(source)]
    edges = [0, *cuts, len(source)]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def spike_grounded(relation: str, spans: list[tuple[int, int]], source: str) -> bool:
    """Both spans in one problem block and a relation cue between/around them."""
    for left, right in problem_blocks(source):
        if all(left <= s < e <= right for s, e in spans):
            lo = min(s for s, _ in spans)
            hi = max(e for _, e in spans)
            window = source[lo:hi].lower()
            return any(cue in window for cue in ACI_RELATION_CONTRACT[relation].get("cues", ()))
    return False


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
            print(f"  attempt {attempt}: {type(error).__name__}; waiting {wait:.0f}s", flush=True)
            if wait:
                time.sleep(wait)
            continue
        if resp.choices and (resp.choices[0].message.content or "").strip():
            return json.loads(resp.choices[0].message.content)
        print(f"  attempt {attempt}: empty completion, immediate retry", flush=True)
    print(f"  gave up after {attempt} attempts / {DRAW_BUDGET_S}s", flush=True)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=2)
    args = parser.parse_args()

    source = json.load(open("/tmp/task_arms_qa_v2_d2n002.json"))["clinical"]["aci/D2N002"]["no_privacy"][0]
    environment = json.load(open("/tmp/qa-v2-d2n002-nemotron-r15.json"))["documents"]["aci/D2N002"]
    inventory = {r["span_label"]: r for r in relation_teacher_span_inventory(environment)}

    base = relation_teacher_prompt("aci/D2N002", source, environment)
    head, sep, tail = base.partition("PRIVACY-SAFE QA")
    prompt = f"{head}{MULTITURN_RULE}\n\n{sep}{tail}"
    response_format = relation_teacher_response_format(environment, source)

    client = OpenAI(base_url=RELATION_TEACHER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"],
                    max_retries=0, timeout=120)

    for n in range(1, args.draws + 1):
        print(f"=== draw {n} ===", flush=True)
        payload = draw(client, prompt, response_format)
        if payload is None:
            continue
        span_pairs = payload.get("span_relations", [])
        rows = []
        for p in span_pairs:
            labels = [a.get("span_label") for a in p.get("arguments") or []]
            if not all(l in inventory for l in labels):
                continue
            spans = [(inventory[l]["start"], inventory[l]["end"]) for l in labels]
            surfaces = [inventory[l]["surface"] for l in labels]
            same_turn = abs(spans[0][0] - spans[1][0]) < 200  # crude: near = old-reachable
            rows.append({
                "relation": p["relation"],
                "pair": f"{surfaces[0]} -> {surfaces[1]}",
                "labels": labels,
                "multiturn": not same_turn,
                "spike_grounded": spike_grounded(p["relation"], spans, source),
            })
        accepted, rejected = compile_relational_assertions(
            "aci/D2N002", source, environment,
            list(span_pairs) + list(payload.get("context_relations", [])),
        )
        print(json.dumps({
            "span_pair_proposals": rows,
            "literal_proposals": len(payload.get("context_relations", [])),
            "production_accepted": [(r["relation"],
                                     [a.get("surface", a.get("literal")) for a in r["evidence"]["arguments"]])
                                    for r in accepted],
            "production_rejected": [r["detail_reason"] for r in rejected],
        }, indent=2), flush=True)


if __name__ == "__main__":
    main()
