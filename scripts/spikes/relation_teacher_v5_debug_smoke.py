"""Debug smoke for the v5 relation-teacher contract on D2N002.

Calls OpenRouter Nemotron directly with the v5 prompt + bound strict schema,
with the reasoning trace INCLUDED in the reply (debug-only; the production
teacher pin excludes it). Prints reasoning, usage, the final JSON, and the
deterministic compile outcome. Raw response is written to the path given by
--raw-out (default: alongside /tmp) for inspection; nothing is written to the
production llm cache because the params differ from the pinned config.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from openai import OpenAI

from cloak.train.qa_builder import (
    RELATION_TEACHER_BASE_URL,
    RELATION_TEACHER_MODEL,
    _validated_candidate_accounting,
    compile_relational_assertions,
    relation_teacher_prompt,
    relation_teacher_response_format,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="/tmp/task_arms_qa_v2_d2n002.json")
    parser.add_argument("--env", default="/tmp/qa-v2-d2n002-nemotron-r15.json")
    parser.add_argument("--doc", default="aci/D2N002")
    parser.add_argument("--raw-out", default="/tmp/relation_teacher_v5_debug_raw.json")
    parser.add_argument("--exclude-reasoning", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    source = json.load(open(args.arms))["clinical"][args.doc]["no_privacy"][0]
    environment = json.load(open(args.env))["documents"][args.doc]

    prompt = relation_teacher_prompt(args.doc, source, environment)
    response_format = relation_teacher_response_format(environment, source)
    print(f"prompt chars: {len(prompt)}", flush=True)

    client = OpenAI(
        base_url=RELATION_TEACHER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_retries=8,
    )
    resp = client.chat.completions.create(
        model=RELATION_TEACHER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format=response_format,
        extra_body={"reasoning": {"exclude": bool(args.exclude_reasoning)}},
        **({"seed": args.seed} if args.seed is not None else {}),
    )
    with open(args.raw_out, "w") as f:
        f.write(resp.model_dump_json(indent=2))
    print(f"raw response -> {args.raw_out}", flush=True)

    choice = resp.choices[0]
    message = choice.message
    reasoning = getattr(message, "reasoning", None)
    if reasoning is None and getattr(message, "model_extra", None):
        reasoning = message.model_extra.get("reasoning")
    usage = resp.usage
    print(f"finish_reason: {choice.finish_reason}")
    if usage is not None:
        details = getattr(usage, "completion_tokens_details", None)
        print(f"usage: prompt={usage.prompt_tokens} completion={usage.completion_tokens}"
              + (f" reasoning={getattr(details, 'reasoning_tokens', None)}" if details else ""))

    print("\n=== REASONING ===")
    print(reasoning if reasoning else "(none returned)")

    content = message.content or ""
    print("\n=== CONTENT ===")
    print(content)

    payload = json.loads(content)
    print("\n=== COMPILE REPLAY ===")
    try:
        ledger = _validated_candidate_accounting(
            payload.get("candidate_accounting", []), environment, source
        )
        print(f"ledger rows: {len(ledger)}")
        for row in ledger:
            print(f"  {row['candidate_label']:>4} {row['state']:<22} {row['reason']}")
    except ValueError as error:
        print(f"ledger INVALID: {error}")
    accepted, rejected = compile_relational_assertions(
        args.doc, source, environment, payload.get("relations", [])
    )
    print(f"accepted: {len(accepted)}")
    for row in accepted:
        print(f"  + {row['relation']} occ={row['occurrence_ids']} q={row['question']!r} a={row['accepted_values']}")
    print(f"rejected: {len(rejected)}")
    for row in rejected:
        print(f"  - proposal[{row['proposal_index']}] {row['detail_reason']}")


if __name__ == "__main__":
    main()
