"""Show verbatim doc_orig -> doc_p under two detectors (gliner_small vs knowledgator) via the
full LatticeCloak Substitutor. Local only (lattice = zero-cost sources; guess-back = MTI probe).

Run: PYTHONPATH=src .venv/bin/python -u scripts/pii_docp_examples.py
"""
import json

from cloak.substitute import Substitutor

DOCS = [
    ("clinical/aci", "corpora/clinical/aci.jsonl", 0),
    ("enron/email", "corpora/enron/replies.jsonl", 0),
    ("synthpai", "corpora/synthpai/train.jsonl", 0),
]
MODEL_A = "urchade/gliner_small-v2.1"
MODEL_B = "knowledgator/gliner-pii-base-v1.0"


def load_text(path, idx):
    with open(path) as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)["text"]


def run(model, texts):
    import torch
    sub = Substitutor(gliner_model=model)
    out = [sub(t) for t in texts]  # (doc_p, R)
    del sub
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main():
    texts = [load_text(p, i) for _, p, i in DOCS]
    A = run(MODEL_A, texts)
    B = run(MODEL_B, texts)
    out_lines = [f"A = {MODEL_A}", f"B = {MODEL_B}", ""]
    for (label, _, _), text, (pa, ra), (pb, rb) in zip(DOCS, texts, A, B):
        out_lines += [
            "#" * 80, f"# {label}  (chars={len(text)})  A_subs={len(ra)}  B_subs={len(rb)}",
            "#" * 80,
            "----- doc_orig -----", text,
            f"----- doc_p [A: gliner_small, {len(ra)} subs] -----", pa,
            f"----- doc_p [B: knowledgator, {len(rb)} subs] -----", pb, ""]
    blob = "\n".join(out_lines)
    open("results/pii_docp_examples.txt", "w").write(blob)
    print(blob)


if __name__ == "__main__":
    main()
