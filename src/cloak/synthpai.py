"""SynthPAI corpus: doc = one author's comment thread + 8 gold profile attributes."""
import json
from pathlib import Path

PATH = Path("corpora/synthpai/train.jsonl")
ATTRS = ["age", "sex", "city_country", "birth_city_country", "education",
         "occupation", "income_level", "relationship_status"]


def load_docs(n: int = 60, min_comments: int = 5) -> list[dict]:
    """Deterministic slice: authors sorted alphabetically, >= min_comments, first n."""
    by_author: dict[str, dict] = {}
    for line in open(PATH):
        r = json.loads(line)
        d = by_author.setdefault(r["author"], {"author": r["author"], "comments": [],
                                               "profile": r["profile"]})
        d["comments"].append(r["text"])
    docs = [d for a, d in sorted(by_author.items()) if len(d["comments"]) >= min_comments][:n]
    for d in docs:
        d["text"] = "\n\n".join(d["comments"])
        d["gold"] = {k: d["profile"].get(k) for k in ATTRS}
    return docs


if __name__ == "__main__":
    docs = load_docs()
    assert len(docs) == 60 and all(len(d["text"]) > 200 for d in docs)
    print(f"60 docs, chars med={sorted(len(d['text']) for d in docs)[30]}, "
          f"gold attrs: {list(docs[0]['gold'])}")
