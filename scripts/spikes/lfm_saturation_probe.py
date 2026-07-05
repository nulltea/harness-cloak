"""10-minute saturation probe for the pinned round-trip reward model (RT_MODEL)
(perf gate prerequisite: the plan's wall-time estimates assume ~1500 RT/h — this
measures the real number). Whatever RT_MODEL currently pins is what gets measured;
generation config (temperature/max_tokens/enable_thinking) matches the roundtrip
module's client exactly, so this is the number the reward path actually pays.

Unique prompts (cache-busting nonce) through the REAL task template at workers 1 and 6.
Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/lfm_saturation_probe.py \
       [--n 24] [--workers 1,6]     # NO cache env var — this must hit the model
"""
import argparse
import json
import time
import uuid

from cloak.corpora import load_task_docs
from cloak.tasks import TASK_TEMPLATE
from cloak.train.roundtrip import MAX_TOKENS, RT_MODEL
from inferdpt.llm import LLMClient
from inferdpt.pipeline import pmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--workers", default="1,6")
    args = ap.parse_args()
    docs = load_task_docs("clinical", max(4, args.n // 4))
    remote = LLMClient(RT_MODEL, temperature=0.0, max_tokens=MAX_TOKENS,
                       extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    for w in [int(x) for x in args.workers.split(",")]:
        prompts = [TASK_TEMPLATE["clinical"].format(doc=docs[i % len(docs)]["text"])
                   + f"\n[probe-nonce {uuid.uuid4()}]" for i in range(args.n)]
        t0 = time.time()
        outs = pmap(remote.generate, prompts, workers=w)
        wall = time.time() - t0
        toks = sum(len(o) for o in outs) / 4          # chars/4 ~ tokens
        print(json.dumps({"workers": w, "n": args.n, "wall_s": round(wall, 1),
                          "rt_per_hour": round(args.n / wall * 3600),
                          "approx_tok_s": round(toks / wall)}), flush=True)


if __name__ == "__main__":
    main()
