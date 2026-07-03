"""P7 (zero-shot extensibility) probe: GLiNER zero-shot recall on held-out entity types
NOT in the TAB-8 schema, as a proxy for user-defined types.

Purpose: measure whether an open-label (GLiNER) head recognizes nameable world-knowledge types
from a label phrase alone. Run pre-fine-tuning (off-the-shelf init) and, later, on the TAB-fine-tuned
Arm B — the pre/post delta is the "generality retention" number that decides whether TAB fine-tuning
preserves the open interface (P6/P7) or narrows it away.

Data: MultiNERD (Babelscape/multinerd) English test, fine-grained types. Held-out set = types
outside TAB's 8 (PER/ORG/LOC/TIME overlap TAB and are excluded; EVE excluded as MISC-adjacent).
Fixed label phrases held constant across checkpoints (honesty: no per-model label tuning).

Run: PYTHONPATH=src .venv/bin/python -u scripts/pii_zeroshot_generality.py --gliner-model <id>
"""
import argparse
import json
import time

# MultiNERD id -> BIO tag (verified against the dataset card).
ID2TAG = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC",
          7: "B-ANIM", 8: "I-ANIM", 9: "B-BIO", 10: "I-BIO", 11: "B-CEL", 12: "I-CEL",
          13: "B-DIS", 14: "I-DIS", 15: "B-EVE", 16: "I-EVE", 17: "B-FOOD", 18: "I-FOOD",
          19: "B-INST", 20: "I-INST", 21: "B-MEDIA", 22: "I-MEDIA", 23: "B-MYTH", 24: "I-MYTH",
          25: "B-PLANT", 26: "I-PLANT", 27: "B-TIME", 28: "I-TIME", 29: "B-VEHI", 30: "I-VEHI"}

# Held-out types (outside TAB-8) -> natural-language label phrase fed to GLiNER zero-shot.
LABELS = {
    "ANIM": "animal",
    "BIO": "biological entity such as a virus, bacterium, or protein",
    "CEL": "celestial body such as a planet, star, or galaxy",
    "DIS": "disease or medical condition",
    "FOOD": "food or drink",
    "INST": "musical instrument or tool",
    "MEDIA": "media title such as a book, song, film, or TV show",
    "MYTH": "mythological or legendary figure",
    "PLANT": "plant",
    "VEHI": "vehicle",
}
PHRASE2TYPE = {v: k for k, v in LABELS.items()}


def build_doc(tokens, tag_ids):
    """Join tokens into text (single spaces); return (text, [(start,end,type)]) gold spans
    for held-out types only."""
    text_parts, offs, pos = [], [], 0
    for t in tokens:
        offs.append(pos)
        text_parts.append(t)
        pos += len(t) + 1  # trailing space
    text = " ".join(tokens)
    spans, cur = [], None  # cur = [start_char, end_char, type]
    for i, tid in enumerate(tag_ids):
        tag = ID2TAG.get(tid, "O")
        typ = tag[2:] if tag != "O" else None
        held = typ in LABELS
        if tag.startswith("B-") and held:
            if cur:
                spans.append(tuple(cur))
            cur = [offs[i], offs[i] + len(tokens[i]), typ]
        elif tag.startswith("I-") and held and cur and cur[2] == typ:
            cur[1] = offs[i] + len(tokens[i])
        else:
            if cur:
                spans.append(tuple(cur))
            cur = None
    if cur:
        spans.append(tuple(cur))
    return text, spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gliner-model", default="urchade/gliner_small-v2.1")
    ap.add_argument("--n", type=int, default=1500, help="English sentences to sample")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="results/pii_zeroshot_generality.json")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("Babelscape/multinerd", split="test", verification_mode="no_checks")
    ds = ds.filter(lambda x: x["lang"] == "en")
    ds = ds.select(range(min(args.n, len(ds))))

    texts, golds = [], []
    for ex in ds:
        text, spans = build_doc(ex["tokens"], ex["ner_tags"])
        if spans:  # keep only sentences with >=1 held-out gold entity
            texts.append(text)
            golds.append(spans)
    print(f"held-out sentences: {len(texts)} (of {len(ds)} en sampled)", flush=True)

    import torch
    from gliner import GLiNER
    model = GLiNER.from_pretrained(args.gliner_model)
    if torch.cuda.is_available():
        model = model.to("cuda")

    label_phrases = list(LABELS.values())
    t0 = time.time()
    preds_per_doc = model.batch_predict_entities(
        texts, label_phrases, threshold=args.threshold, batch_size=args.batch_size)

    from collections import defaultdict
    hit_any, hit_typed, tot = defaultdict(int), defaultdict(int), defaultdict(int)
    n_pred = n_pred_on_gold = 0
    for spans, ents in zip(golds, preds_per_doc):
        preds = [(e["start"], e["end"], PHRASE2TYPE[e["label"]]) for e in ents]
        n_pred += len(preds)
        # any-gold overlap (precision proxy over all held-out golds in this sentence)
        for ps, pe, pt in preds:
            if any(ps < ge and gs < pe for gs, ge, _ in spans):
                n_pred_on_gold += 1
        for gs, ge, gt in spans:
            tot[gt] += 1
            over = [(ps, pe, pt) for ps, pe, pt in preds if ps < ge and gs < pe]
            if over:
                hit_any[gt] += 1
            if any(pt == gt for _, _, pt in over):
                hit_typed[gt] += 1

    types = sorted(tot)
    per_type = {t: {"any": hit_any[t] / tot[t], "typed": hit_typed[t] / tot[t], "n": tot[t]}
                for t in types}
    T = sum(tot.values())
    res = {
        "gliner_model": args.gliner_model, "dataset": "multinerd/en/test",
        "threshold": args.threshold, "n_sentences": len(texts), "n_gold": T,
        "overall": {"any": sum(hit_any.values()) / T, "typed": sum(hit_typed.values()) / T},
        "precision_proxy": n_pred_on_gold / max(n_pred, 1), "n_pred": n_pred,
        "per_type": per_type, "wall_s": round(time.time() - t0, 1),
    }
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps({"model": args.gliner_model, "overall": res["overall"],
                      "precision_proxy": round(res["precision_proxy"], 3),
                      "per_type_any": {t: round(per_type[t]["any"], 3) for t in types}}, indent=2))


def _selfcheck():
    # tokens with a 2-token ANIM span and a 1-token FOOD span
    toks = ["The", "polar", "bear", "ate", "sushi", "."]
    tags = [0, 7, 8, 0, 17, 0]  # B-ANIM I-ANIM ... B-FOOD
    text, spans = build_doc(toks, tags)
    assert text == "The polar bear ate sushi .", text
    got = {(text[s:e], t) for s, e, t in spans}
    assert got == {("polar bear", "ANIM"), ("sushi", "FOOD")}, got
    print("selfcheck OK")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
