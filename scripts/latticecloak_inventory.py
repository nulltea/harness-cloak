"""P1.1 entity inventory: route detected spans on 60 SynthPAI docs, list teacher-needed entities.

Routes (plan P1): PERSON/CODE -> placeholder (direct); DATETIME/QUANTITY -> rule bucket;
LOC -> geonames; other types whose head lemma is in WordNet -> wordnet; else -> teacher.

Run: PYTHONPATH=src .venv/bin/python -u scripts/latticecloak_inventory.py
"""
import json
import time
from collections import Counter, defaultdict

from cloak.detect import Detector, coref_chains
from cloak.synthpai import load_docs

DIRECT = {"PERSON": "placeholder", "CODE": "placeholder"}
BUCKET = {"DATETIME": "bucket", "QUANTITY": "bucket"}


def wordnet_has(phrase: str) -> bool:
    from nltk.corpus import wordnet as wn
    p = phrase.lower().strip()
    return bool(wn.synsets(p.replace(" ", "_")) or
                (len(p.split()) > 1 and wn.synsets(p.split()[-1])))


def route(span) -> str:
    if span.type in DIRECT:
        return DIRECT[span.type]
    if span.type in BUCKET:
        return BUCKET[span.type]
    if span.type == "LOC":
        return "geonames"
    return "wordnet" if wordnet_has(span.text) else "teacher"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gliner-model", default="urchade/gliner_small-v2.1")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="data/latticecloak_teacher_entities.json")
    args = ap.parse_args()

    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    docs = load_docs(args.n)
    det = Detector(gliner_model=args.gliner_model)
    routes = Counter()
    teacher = defaultdict(lambda: {"count": 0, "types": Counter(), "context": ""})
    t0 = time.time()

    for i, doc in enumerate(docs):
        spans = coref_chains(doc["text"], det.detect(doc["text"]))
        # canonical form per chain = longest mention text among the chain's spans
        canon = {}
        for s in spans:
            if s.chain >= 0:
                cur = canon.get(s.chain, "")
                canon[s.chain] = s.text if len(s.text) > len(cur) else cur
        for s in spans:
            r = route(s)
            routes[r] += 1
            if r == "teacher":
                key = canon.get(s.chain, s.text).lower() if s.chain >= 0 else s.text.lower()
                e = teacher[key]
                e["count"] += 1
                e["types"][s.type] += 1
                if not e["context"]:
                    e["context"] = doc["text"][max(0, s.start - 80):s.end + 40].replace("\n", " ")
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(docs)}] spans so far: {sum(routes.values())} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    out = {
        "docs": len(docs), "gliner_model": args.gliner_model, "route_counts": dict(routes),
        "teacher_unique": len(teacher),
        "entities": {k: {"count": v["count"], "types": dict(v["types"]), "context": v["context"]}
                     for k, v in sorted(teacher.items(), key=lambda kv: -kv[1]["count"])},
        "wall_s": round(time.time() - t0, 1),
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("route_counts", "teacher_unique", "wall_s")}, indent=2))
    print("top teacher entities:",
          [k for k in list(out["entities"])[:12]])


if __name__ == "__main__":
    main()
