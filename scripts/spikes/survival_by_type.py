"""Per-type 'survived span' estimate via a grounded LLM alignment judge.

Replaces the fuzzy>=60 'present_in_out_p' denominator (extractor_pointer_by_type.py),
which counts generic-fill fuzzy noise as present (~87% of the band60_90 bucket is noise;
see research-wiki/experiments/extractor-pointer-by-type.md and the miss audit).

A generalization span 'survived' iff its fill's content is genuinely mentioned in out_p
(exact OR reworded), so an inverter could locate and specialize it. Method:

  1. Deterministic anchor: fill exact word-boundary match in out_p -> SURVIVED (certain).
     These double as calibration ground truth (judge-vs-exact agreement, free).
  2. LLM alignment judge (Qwen3.6 teacher, temp 0, non-thinking; a DIFFERENT family from
     gemma which generated out_p -> no self-preference). One call per doc, all spans
     batched, structured JSON: per span {label, quote}.
       SURVIVED  fill present ~verbatim     REWORDED  concept present, different words
       TEMPLATED model emitted a [Bracket]/placeholder refusal, not real content
       ABSENT    not in out_p
  3. Grounding verification (the robustness anchor): SURVIVED/REWORDED must quote a
     verbatim substring of out_p; ungrounded quote -> downgrade to ABSENT. Kills the
     LLM's confident-false-positive failure mode.

survived-for-extraction = SURVIVED + REWORDED. TEMPLATED/ABSENT reported separately.

Run (out_p already cached from prior audit roundtrips):
  INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
    .venv/bin/python -u scripts/spikes/survival_by_type.py \
    --env data/ranker_env.json --arms data/task_arms_tau0.02.json \
    --corpora clinical --n-docs 16
"""
import argparse
import json
import re
from pathlib import Path

from build_arms_artifact import load_artifact
from train_ranker import assemble, derive_spans, floor_walk_choice

from cloak.corpora import load_task_docs
from cloak.train.ranker import CORPORA
from cloak.train.roundtrip import roundtrip_batch

OUT = Path("results/survival_by_type.json")
JUDGE_MODEL = "Qwen3.6-35B-A3B"
JUDGE_BASE_URL = "http://localhost:8060/v1"
PROMPT_VERSION = 1  # bump on any prompt change (part of the LLM cache key via prompt text)
TYPES = ["PERSON", "ORG", "LOC", "DATETIME", "CODE", "QUANTITY", "DEM", "MISC"]

SYSTEM = (
    "You align a rewritten document's answer back to the substitutions made in it. "
    "You are precise and never invent text. Output only JSON."
)

JUDGE_TMPL = """A document was privacy-rewritten: some spans were replaced by more general \
phrases (fills). The rewritten document was sent to another model, which produced an ANSWER.

For each substitution below, decide whether the fill's *content* appears in the ANSWER, so \
that a reader could map it back to a specific span. Labels:
- SURVIVED: the fill appears (near-)verbatim in the ANSWER.
- REWORDED: the same generalized concept appears but with different wording.
- TEMPLATED: the model emitted a bracketed placeholder like [Disease] or [Name] instead of \
real content (a refusal, not a real mention).
- ABSENT: the fill's content does not appear in the ANSWER.

For SURVIVED/REWORDED/TEMPLATED you MUST copy a short verbatim quote (<=10 words) from the \
ANSWER that is the mention. For ABSENT set quote to null.

Original span -> fill (what to look for), with type:
{items}

ANSWER:
\"\"\"{out_p}\"\"\"

Reply with ONLY a JSON array, one object per id in order:
[{{"id": 0, "label": "SURVIVED|REWORDED|TEMPLATED|ABSENT", "quote": "..."|null}}, ...]"""


def _judge():
    from inferdpt.llm import LLMClient
    return LLMClient(JUDGE_MODEL, base_url=JUDGE_BASE_URL, api_key="x", temperature=0.0,
                     max_tokens=1500, extra_body={"chat_template_kwargs": {"enable_thinking": False}})


def exact_present(fill: str, out_p: str) -> bool:
    return bool(re.search(rf"\b{re.escape(fill)}\b", out_p, re.IGNORECASE))


def fill_present(fill: str, out_p: str) -> bool:
    """Fill reached out_p verbatim or as a >=90 fuzzy mention (the substituted form)."""
    if exact_present(fill, out_p):
        return True
    from rapidfuzz import fuzz
    al = fuzz.partial_ratio_alignment(fill.lower(), out_p.lower())
    return bool(al and al.score >= 90.0)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def grounded(quote, out_p: str) -> bool:
    return bool(quote) and _norm(quote) in _norm(out_p)


def parse_judge(reply: str, n: int) -> list[dict]:
    m = re.search(r"\[.*\]", reply, re.DOTALL)
    if not m:
        return [{} for _ in range(n)]
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [{} for _ in range(n)]
    by_id = {o.get("id"): o for o in arr if isinstance(o, dict)}
    return [by_id.get(i, {}) for i in range(n)]


def build_jobs(args):
    art = load_artifact(args.arms)
    env = json.loads(Path(args.env).read_text())
    jobs, metas = [], []
    for corpus in args.corpora.split(","):
        texts = {d["id"]: d["text"] for d in load_task_docs(corpus, args.n_docs)}
        for doc_id, d in env["corpora"].get(corpus, {}).items():
            if not d.get("spans") or doc_id not in texts:
                continue
            # feats are discarded here, so the corpus one-hot is irrelevant; pass an
            # in-list corpus to satisfy action_features for corpora added after CORPORA
            # was frozen (e.g. lexsum). The real corpus drives load_task_docs + the task
            # template above/below.
            fc = corpus if corpus in CORPORA else CORPORA[0]
            spans, _ = derive_spans(d["spans"], dict(env["k_floors"]), fc, "cpu")
            choice = floor_walk_choice(spans)
            doc_p, R = assemble(texts[doc_id], art[corpus][doc_id]["tau_walk"][1],
                                d["spans"], choice)
            jobs.append({"corpus": corpus, "doc_p": doc_p, "R": R, "probes": []})
            metas.append({"corpus": corpus, "doc_id": doc_id, "R": R, "doc_p": doc_p})
    return jobs, metas


def new_row():
    return {"substituted": 0, "SURVIVED": 0, "REWORDED": 0, "TEMPLATED": 0, "ABSENT": 0,
            "leaked_only": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="data/ranker_env.json")
    ap.add_argument("--arms", default="data/task_arms_tau0.02.json")
    ap.add_argument("--corpora", default="clinical")
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    jobs, metas = build_jobs(args)
    outs = roundtrip_batch(jobs, workers=args.workers)
    judge = _judge()

    rows = {t: new_row() for t in TYPES}
    calib = {"exact_total": 0, "judge_agree": 0}  # judge on exact-match spans (free truth)
    ground = {"downgraded": 0, "judge_positive": 0}
    examples = []

    for m, o in zip(metas, outs):
        out_p = o["out_p"]
        gens = [e for e in m["R"] if e["action"] == "generalize"]
        if not gens:
            continue
        items = "\n".join(
            f'{i}. "{e["surface"]}" -> "{e["replacement"]}"  [{e.get("type", "MISC")}]'
            for i, e in enumerate(gens))
        reply = judge.generate(JUDGE_TMPL.format(items=items, out_p=out_p), system=SYSTEM)
        verdicts = parse_judge(reply, len(gens))

        for e, v in zip(gens, verdicts):
            typ = e.get("type", "MISC")
            if typ not in rows:
                typ = "MISC"
            rows[typ]["substituted"] += 1

            label = v.get("label", "ABSENT")
            quote = v.get("quote")
            if label in ("SURVIVED", "REWORDED", "TEMPLATED"):
                ground["judge_positive"] += 1
            # grounding: a claimed mention must be a real substring of out_p
            if label in ("SURVIVED", "REWORDED") and not grounded(quote, out_p):
                label = "ABSENT"
                ground["downgraded"] += 1

            # deterministic anchor overrides: exact match is certain SURVIVED
            is_exact = exact_present(e["replacement"], out_p)
            if is_exact:
                calib["exact_total"] += 1
                if v.get("label") in ("SURVIVED", "REWORDED"):
                    calib["judge_agree"] += 1
                label = "SURVIVED"

            if label not in rows[typ]:
                label = "ABSENT"
            rows[typ][label] += 1

            # LEAKED-ONLY guard: the judge may credit 'survived' to a span whose surviving
            # text is the ORIGINAL surface (an undetected duplicate leaked through doc_p),
            # while the substituted fill never reached out_p. That is a privacy leak, NOT the
            # substituted span surviving — flag it so the substituted-content count excludes it.
            if label in ("SURVIVED", "REWORDED") and not fill_present(e["replacement"], out_p) \
                    and exact_present(e["surface"], out_p):
                rows[typ]["leaked_only"] += 1

            if not is_exact and label in ("SURVIVED", "REWORDED") and len(examples) < 25:
                examples.append({"doc": m["doc_id"], "type": typ, "surface": e["surface"],
                                 "fill": e["replacement"], "label": label, "quote": quote})

    def survived(r):  # judge-credited survival (SURVIVED+REWORDED)
        return r["SURVIVED"] + r["REWORDED"]

    def subst_survived(r):  # substituted content actually reached out_p (leak excluded)
        return survived(r) - r["leaked_only"]

    out_rows = []
    for t in TYPES:
        r = rows[t]
        n = r["substituted"]
        out_rows.append({"type": t, **r, "survived": survived(r),
                         "subst_survived": subst_survived(r),
                         "subst_survival_rate": round(subst_survived(r) / n, 3) if n else None})
    tot = new_row()
    for r in rows.values():
        for k in tot:
            tot[k] += r[k]
    n = tot["substituted"]
    totals = {**tot, "survived": survived(tot), "subst_survived": subst_survived(tot),
              "subst_survival_rate": round(subst_survived(tot) / n, 3) if n else None}

    report = {
        "settings": {**vars(args), "judge_model": JUDGE_MODEL, "prompt_version": PROMPT_VERSION},
        "n_docs": len(metas), "rows": out_rows, "totals": totals,
        "calibration_judge_vs_exact": {
            **calib,
            "agreement": round(calib["judge_agree"] / calib["exact_total"], 3)
            if calib["exact_total"] else None},
        "grounding": {**ground,
                      "downgrade_rate": round(ground["downgraded"] / ground["judge_positive"], 3)
                      if ground["judge_positive"] else None},
        "reworded_survived_examples": examples,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k != "reworded_survived_examples"}, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
