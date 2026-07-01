"""Build row-aligned φ caches on the whole-word sub-vocab for the Lever-2 A/B.

sub-vocab = data/vocab.json (12k cl100k tokens) ∩ counter-fitted ∩ glove vocab, in
vocab.json order. Builds three caches over the SAME word list (fair A/B):
  data/vocab_cf      counter-fitted (P2-strong: synonym retrofit)
  data/vocab_glove   glove/paragram (P2-weak: same vocab/dim, no retrofit) — P2 isolation
  data/vocab_qwen_sub  qwen3-embedding-0.6b re-embedded on the sub-vocab (baseline)

Run: PYTHONPATH=src python scripts/build_phi_subvocab.py --cf <cf.txt> --glove <glove.txt>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inferdpt.embeddings import (VocabEmbeddings, build_from_static_vectors,
                                 embed, load_static_vectors)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="data/vocab.json")
    ap.add_argument("--cf", required=True, help="counter-fitted-vectors.txt")
    ap.add_argument("--glove", required=True, help="glove.txt (same vocab as cf)")
    ap.add_argument("--no-qwen", action="store_true", help="skip the API re-embed (offline)")
    args = ap.parse_args()

    full = json.loads(Path(args.vocab).read_text())
    cf_tab = load_static_vectors(args.cf)
    gl_tab = load_static_vectors(args.glove)
    sub = [w for w in full if w in cf_tab and w in gl_tab]
    print(f"full={len(full)}  cf_vocab={len(cf_tab)}  sub-vocab={len(sub)} "
          f"({100*len(sub)/len(full):.0f}% coverage)")

    cf = build_from_static_vectors(sub, args.cf, "data/vocab_cf")
    gl = build_from_static_vectors(sub, args.glove, "data/vocab_glove")
    assert cf.vocab == gl.vocab == sub, "cf/glove row alignment broke"
    print(f"built cf {cf.matrix.shape} Δφ_mean={cf.sensitivity.mean():.3f} | "
          f"glove {gl.matrix.shape} Δφ_mean={gl.sensitivity.mean():.3f}")

    if not args.no_qwen:
        M = embed(sub)  # served qwen3-embedding-0.6b
        M /= (M ** 2).sum(1, keepdims=True) ** 0.5 + 1e-8
        VocabEmbeddings(sub, M.astype("float32")).save("data/vocab_qwen_sub")
        print(f"built qwen_sub [{len(sub)}, {M.shape[1]}]")


if __name__ == "__main__":
    main()
