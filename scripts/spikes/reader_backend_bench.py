"""Reader-backend speed smoke: served-Q8 (serial, llama.cpp prompt-cache) vs local fla
(torch-conv shim, prefix-KV cross-batch reuse). Same (context, questions) workload.

Real-generation timing: reader responses for these docs are already disk-cached from the
re-gate, so each question gets a unique nonce (in the QUESTION suffix, not the note) to force
cache misses while keeping the note prefix shared — so llama.cpp's prompt-cache (A) and
_read_prefix (B) still reuse it. gemma contexts come from the warm cache (fast).

A: served Qwen3.5-0.8B, workers=1 sequential to one slot (cache_prompt reuses the note KV).
B: local Qwen3.5-0.8B bf16, torch-conv shim -> fla fast path, _read_prefix (encode note once,
   batch-expand KV, decode question suffixes); falls back to batched generate if the hybrid
   cache won't batch-expand.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
       scripts/spikes/reader_backend_bench.py [--per-corpus 2]
"""
import argparse, json, time
from pathlib import Path

import torch, torch.nn.functional as F
from cloak.corpora import load_task_docs
from cloak.train.roundtrip import RT_BASE_URL, roundtrip_batch

CORPORA = ("clinical", "lexsum", "wikibio")
PROMPT = ("Answer the question using ONLY the note below. Reply with the shortest exact answer "
          "copied from the note. If not present, reply NONE.\n\nNote:\n{ctx}\n\nQuestion: {q}\nAnswer:")


def parse(r):
    a = r.strip().splitlines()[0].strip() if r and r.strip() else ""
    return "" if a.upper().strip(".:` ") == "NONE" else a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-corpus", type=int, default=2)
    args = ap.parse_args()
    env = json.loads(Path("data/ranker_env_full.json").read_text())["corpora"]

    docs = []
    for corpus in CORPORA:
        texts = {x["id"]: x["text"] for x in load_task_docs(corpus)}
        for did, d in [(i, x) for i, x in env[corpus].items()
                       if x.get("spans") and x.get("probes", {}).get("train")][:args.per_corpus]:
            ctx = roundtrip_batch([{"corpus": corpus, "doc_p": texts[did], "R": [], "probes": []}],
                                  workers=1)[0]["out_final"]
            # nonce in the question suffix -> forces disk-cache miss, note prefix stays shared
            qs = [f"{p['question']} (q{i})" for i, p in enumerate(d["probes"]["train"])]
            docs.append((did, ctx, qs))
    nq = sum(len(q) for _, _, q in docs)
    print(f"docs={len(docs)} questions={nq}", flush=True)

    # ---------- A: served serial (llama.cpp prompt-cache) ----------
    from inferdpt.llm import LLMClient
    cli = LLMClient("Qwen3.5-0.8B", base_url=RT_BASE_URL, api_key="x", temperature=0.0,
                    max_tokens=32, extra_body={"chat_template_kwargs": {"enable_thinking": False},
                                               "cache_prompt": True})
    t = time.time(); Aans = []
    for _, ctx, qs in docs:
        for q in qs:
            Aans.append(parse(cli.generate(PROMPT.format(ctx=ctx, q=q))))
    Awall = time.time() - t
    print(f"\nA served-serial-cache : {Awall:6.1f}s  ({Awall/nq*1000:5.0f} ms/q)", flush=True)

    # ---------- B: local fla (shim) + _read_prefix ----------
    import transformers.models.qwen3_5.modeling_qwen3_5 as mod
    mod.causal_conv1d_fn = lambda x, weight, bias=None, activation=None, seq_idx=None: (
        F.silu(F.conv1d(x, weight.unsqueeze(1), bias, padding=weight.shape[1]-1,
                        groups=weight.shape[0])[..., :x.shape[-1]])
        if activation in ("silu", "swish")
        else F.conv1d(x, weight.unsqueeze(1), bias, padding=weight.shape[1]-1,
                      groups=weight.shape[0])[..., :x.shape[-1]])
    mod.causal_conv1d_update = mod.torch_causal_conv1d_update
    mod.is_fast_path_available = True
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16).to("cuda").eval()

    def prompts_of(ctx, qs):
        return [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(ctx=ctx, q=q)}],
                add_generation_prompt=True, enable_thinking=False, tokenize=False) for q in qs]

    def batched(ctx, qs):
        enc = tok(prompts_of(ctx, qs), return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)
        return [parse(r) for r in tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]

    def prefix(ctx, qs):
        ids = [tok(p, add_special_tokens=False).input_ids for p in prompts_of(ctx, qs)]
        m0 = ids[0]; L = 0
        while L < len(m0) and all(L < len(e) and e[L] == m0[L] for e in ids): L += 1
        L = min(L, min(len(e) for e in ids) - 1)
        if L <= 0: raise ValueError("no shared prefix")
        past = m(torch.tensor([m0[:L]], device="cuda"), use_cache=True).past_key_values
        past.batch_repeat_interleave(len(qs))   # raises if hybrid cache can't
        suf = [e[L:] for e in ids]; S = max(len(s) for s in suf); pad = tok.pad_token_id
        inp = torch.tensor([[pad]*(S-len(s))+s for s in suf], device="cuda")
        sm = torch.tensor([[0]*(S-len(s))+[1]*len(s) for s in suf], device="cuda")
        attn = torch.cat([torch.ones(len(qs), L, dtype=torch.long, device="cuda"), sm], 1)
        pos = (attn.long().cumsum(-1)-1).clamp(min=0)[:, L:]
        with torch.no_grad():
            out = m.generate(input_ids=inp, attention_mask=attn, past_key_values=past,
                             position_ids=pos, max_new_tokens=32, do_sample=False, pad_token_id=pad)
        return [parse(r) for r in tok.batch_decode(out[:, inp.shape[1]:], skip_special_tokens=True)]

    # warm (JIT compile) once, untimed
    _ = batched(docs[0][1], docs[0][2][:2])
    mode = "prefix"
    try:
        _ = prefix(docs[0][1], docs[0][2][:2])
    except Exception as e:
        mode = "batched"; print(f"  (_read_prefix unusable on hybrid cache: {type(e).__name__}: {e} -> batched)", flush=True)
    fn = prefix if mode == "prefix" else batched
    torch.cuda.synchronize(); t = time.time(); Bans = []
    for _, ctx, qs in docs:
        Bans += fn(ctx, qs)
    torch.cuda.synchronize(); Bwall = time.time() - t
    print(f"B local-fla-{mode:<7}   : {Bwall:6.1f}s  ({Bwall/nq*1000:5.0f} ms/q)", flush=True)

    agree = sum(a == b for a, b in zip(Aans, Bans))
    print(f"\nA vs B answer agreement: {agree}/{nq}  | speedup A/B = {Awall/max(Bwall,1e-6):.2f}x")
    for (a, b) in list(zip(Aans, Bans))[:5]:
        print(f"  A={a[:30]!r:<34} B={b[:30]!r}")


if __name__ == "__main__":
    main()
