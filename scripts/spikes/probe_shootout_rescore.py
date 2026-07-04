"""Re-score the probe-shootout items with a different attacker model (labels only).

Probe scores (p2/p3/p4/p6) are reused from results/privacy_probe_shootout.json — only the
attacker ground truth is recomputed, so swapping referees costs one call per item and zero
local compute. Written after the gemini max_tokens truncation burned the first label set.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/spikes/probe_shootout_rescore.py --attacker "Qwen3.6-35B-A3B"
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.append(str(Path(__file__).resolve().parent))
from privacy_probe_shootout import ATTACK_PROMPT, auc, per_span_rank_agreement  # noqa: E402

from inferdpt.llm import LLMClient  # noqa: E402
from inferdpt.pipeline import pmap  # noqa: E402

LOCAL_URL = "http://localhost:8060/v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacker", required=True)
    ap.add_argument("--remote", action="store_true",
                    help="PAID proxy model — only with explicit permission; throttled 25/min")
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="thinking models bill hidden reasoning: gemini needs ~2000")
    args = ap.parse_args()

    t0 = time.time()
    path = Path("results/privacy_probe_shootout.json")
    report = json.loads(path.read_text())
    items = report["items"]

    from rapidfuzz import fuzz
    if args.remote:
        llm = LLMClient(args.attacker, temperature=0.0, max_tokens=args.max_tokens)

        def ask(it):  # 25 req/min upstream quota
            import openai
            for _ in range(6):
                try:
                    r = llm.generate(ATTACK_PROMPT.format(
                        fill=it["fill"], typ=it["type"], sent=it["sent_p"]))
                    time.sleep(2.6)
                    return r
                except (openai.RateLimitError, openai.InternalServerError,
                        openai.APIConnectionError):  # 429 quota + transient 5xx
                    time.sleep(65)
            return ""
        replies = pmap(ask, items, workers=1)
    else:
        llm = LLMClient(args.attacker, temperature=0.0, max_tokens=args.max_tokens,
                        base_url=LOCAL_URL, api_key="x",
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        replies = pmap(lambda it: llm.generate(ATTACK_PROMPT.format(
            fill=it["fill"], typ=it["type"], sent=it["sent_p"])), items, workers=6)
    truncated = 0
    for it, r in zip(items, replies):
        m = re.search(r"\[.*?\]", r or "", re.DOTALL)
        try:
            guesses = [str(g) for g in json.loads(m.group())][:5] if m else []
        except json.JSONDecodeError:
            guesses = []
        if not guesses:
            truncated += 1
        scores = [fuzz.token_sort_ratio(g.lower(), it["surface"].lower()) for g in guesses]
        it["hit1"] = int(bool(scores) and scores[0] >= 85)
        it["hit5"] = int(any(s >= 85 for s in scores))
        it["guesses"] = guesses
    h1, h5 = mean(i["hit1"] for i in items), mean(i["hit5"] for i in items)
    print(f"attacker {args.attacker}: hit@1={h1:.3f} hit@5={h5:.3f} "
          f"unparsed={truncated}/{len(items)} {time.time()-t0:.0f}s", flush=True)

    probes = {}
    for p in ("p2", "p3", "p4", "p6"):
        probes[p] = {"auc_hit1": auc([i[p] for i in items], [i["hit1"] for i in items]),
                     "auc_hit5": auc([i[p] for i in items], [i["hit5"] for i in items]),
                     "per_span_level_agreement": per_span_rank_agreement(items, p)}
        print(p, probes[p], flush=True)

    report.setdefault("attackers", {})[args.attacker] = {
        "hit1": round(h1, 3), "hit5": round(h5, 3), "unparsed": truncated, "probes": probes}
    report["items"] = items
    path.write_text(json.dumps(report, indent=1))
    print(f"wall {time.time()-t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
