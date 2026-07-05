"""tau sweep on the smoke slice: leakage + utility probes per operating point.

Same probe instruments as the InferDPT report (overlap, pii_leak; SimCSE utility),
so the two pages read on the same scales. Per tau: substitute -> remote out_p
(cached) -> rule-extractor inversion -> out_final -> probes. Also records the knob mechanics: how many
generalized spans sit at the most-specific vs most-general lattice level.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/latticecloak_tau_sweep.py
"""
import argparse
import json
import re
import time
from statistics import mean

from cloak.detect import Detector
from cloak.extract import invert
from cloak.probe import _pipe
from cloak.substitute import substitute
from cloak.synthpai import load_docs
from cloak.tasks import doc_tasks, fill, qa_pairs
from inferdpt.llm import LLMClient
from inferdpt.pipeline import pmap
from inferdpt.probes.leakage import overlap, pii_leakage
from inferdpt.probes.utility import cosine_pairs


def mlm_guess_back(doc_p: str, R: list[dict], top_k: int = 50) -> float:
    """Span-inversion analog of inversion@k: the attacker reads doc_p (replacement visible);
    the slot is masked with the generalization kept as an appositive ("lives in <mask>,
    a city in Norway,") and roberta-base guesses the original. Candidate-sensitive:
    more specific lattice levels give the attacker more signal."""
    fill_pipe = _pipe()
    hits = tries = 0
    for e in R:
        if e["action"] != "generalize":
            continue
        pat = re.compile(re.escape(e["replacement"]), re.IGNORECASE)
        sent = next((s for s in re.split(r"(?<=[.!?\n])\s+", doc_p) if pat.search(s)), None)
        if sent is None:
            continue
        masked = pat.sub(f"{fill_pipe.tokenizer.mask_token}, {e['replacement']},", sent, count=1)
        try:
            preds = fill_pipe(masked, top_k=top_k)
        except Exception:
            continue
        targets = {t.lower() for t in re.findall(r"\w+", e["surface"]) if len(t) > 2} \
            or {e["surface"].lower()}
        tries += 1
        hits += any(p["token_str"].strip().lower() in targets for p in preds)
    return hits / tries if tries else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--taus", default="0.005,0.02,0.1,0.5")
    ap.add_argument("--gen-model", default="gemma 4 (E4B)")  # = roundtrip.RT_MODEL pin (2026-07-05)
    ap.add_argument("--out", default="results/latticecloak_tau_sweep.json")
    args = ap.parse_args()
    taus = [float(t) for t in args.taus.split(",")]

    docs = load_docs(args.limit)
    qa = qa_pairs(docs)
    det = Detector()
    spans_per_doc = {d["author"]: det.detect(d["text"]) for d in docs}
    remote = LLMClient(args.gen_model, temperature=0.0, max_tokens=400,
                       extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    # out_ref is tau-independent: one call per (doc, task)
    ref_rows = [(d, t) for d in docs for t in doc_tasks(d, qa)]
    outs_ref = pmap(lambda dt: remote.generate(fill(dt[1], dt[0]["text"])), ref_rows, workers=8)
    ref = {(d["author"], t["task_id"]): o for (d, t), o in zip(ref_rows, outs_ref)}

    sweep = []
    for tau in taus:
        t0 = time.time()
        subbed = {}
        lvl_specific = lvl_floor = n_gen = 0
        for d in docs:
            import copy
            doc_p, R = substitute(d["text"], copy.deepcopy(spans_per_doc[d["author"]]), tau=tau)
            subbed[d["author"]] = (doc_p, R)
            for e in R:
                if e["action"] == "generalize" and e.get("lattice"):
                    n_gen += 1
                    lvl_specific += e["replacement"].lower() == e["lattice"][0].lower()
                    lvl_floor += e["replacement"].lower() == e["lattice"][-1].lower()
        rows = [(d, t) for d in docs for t in doc_tasks(d, qa)]
        outs_p = pmap(lambda dt: remote.generate(fill(dt[1], subbed[dt[0]["author"]][0])),
                      rows, workers=8)
        finals = [invert(op, subbed[d["author"]][1])[0] for (d, t), op in zip(rows, outs_p)]

        import math
        leak_overlap = mean(overlap(d["text"], subbed[d["author"]][0]) for d in docs)
        pii = [pii_leakage(d["text"], subbed[d["author"]][0])["recall"] for d in docs]
        leak_pii = mean(v for v in pii if not math.isnan(v))
        leak_mlm = mean(mlm_guess_back(*subbed[d["author"]]) for d in docs)
        util_ctrl = mean(cosine_pairs(
            [(ref[(d["author"], t["task_id"])], f) for (d, t), f in zip(rows, finals)]))
        util_p_ctrl = mean(cosine_pairs(
            [(ref[(d["author"], t["task_id"])], op) for (d, t), op in zip(rows, outs_p)]))
        coh_gen_p = mean(cosine_pairs(
            [(subbed[d["author"]][0], op) for (d, t), op in zip(rows, outs_p)]))
        coh_doc_p = mean(cosine_pairs([(d["text"], subbed[d["author"]][0]) for d in docs]))
        sweep.append({"tau": tau, "overlap": round(leak_overlap, 4), "pii_leak": round(leak_pii, 4),
                      "mlm_guess_back": round(leak_mlm, 4),
                      "utility_control": round(util_ctrl, 4),
                      "utility_p_control": round(util_p_ctrl, 4),
                      "coherence_gen_p": round(coh_gen_p, 4), "coherence_doc_p": round(coh_doc_p, 4),
                      "gen_spans": n_gen, "at_most_specific": lvl_specific, "at_floor": lvl_floor,
                      "wall_s": round(time.time() - t0, 1)})
        print(sweep[-1], flush=True)

    json.dump({"docs": len(docs), "tuples": len(ref_rows), "gen_model": args.gen_model,
               "sweep": sweep}, open(args.out, "w"), indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
