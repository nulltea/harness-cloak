"""Embedding vocabulary cache for RANTEXT's φ(·).

φ is computed once, offline, over the English vocab and cached as:
  - `<name>.json`  : the ordered list of vocab token strings
  - `<name>.npy`   : float32 matrix [|V|, dim] of unit-normalised embeddings

The runtime perturbation reads only this cache (no embedding calls at perturb time).
Embeddings come from an OpenAI-compatible `/v1/embeddings` endpoint — the proxy does
not expose one, so this defaults to the local llama-swap container.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The Tailscale proxy does NOT proxy /v1/embeddings; llama-swap (host-net :8060) does.
EMBED_BASE_URL = "http://localhost:8060/v1"
EMBED_MODEL = "qwen3-embedding-0.6b"


def embed(texts: list[str], *, model: str = EMBED_MODEL, base_url: str = EMBED_BASE_URL,
          batch: int = 256) -> np.ndarray:
    """Embed `texts`, returning a float32 [n, dim] array (in input order)."""
    from openai import OpenAI  # lazy: only the served-API embed path needs openai

    client = OpenAI(base_url=base_url, api_key="not-needed")
    vecs: list[list[float]] = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=model, input=texts[i:i + batch])
        vecs.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return np.asarray(vecs, dtype=np.float32)


@dataclass
class VocabEmbeddings:
    """Loaded vocab + embedding matrix, with O(1) token→row lookup."""

    vocab: list[str]
    matrix: np.ndarray  # [|V|, dim], float32

    def __post_init__(self) -> None:
        self.index = {w: i for i, w in enumerate(self.vocab)}

    @property
    def sensitivity(self) -> np.ndarray:
        """Δφ: per-dimension coordinate range (max−min over V), shape [dim] — the φ
        sensitivity vector. Each dim calibrates its own Laplace scale δ_dim/Z(ε),
        matching the reference (`delta_f_new` in mengtong0110/InferDPT func.py).
        NOT a scalar max: collapsing to max_dim δ_dim inflates the random radius by
        δ_max/RMS(δ) → |C_r| saturates to 100% → word salad."""
        return (self.matrix.max(0) - self.matrix.min(0)).astype(np.float64)

    @classmethod
    def load(cls, name: str | Path) -> "VocabEmbeddings":
        name = Path(name)
        vocab = json.loads(name.with_suffix(".json").read_text())
        matrix = np.load(name.with_suffix(".npy"))
        return cls(vocab, matrix)

    def save(self, name: str | Path) -> None:
        name = Path(name)
        name.parent.mkdir(parents=True, exist_ok=True)
        name.with_suffix(".json").write_text(json.dumps(self.vocab))
        np.save(name.with_suffix(".npy"), self.matrix)


def build(out: str | Path, *, limit: int = 12000, tokeniser_path: str | None = None,
          model: str = EMBED_MODEL, base_url: str = EMBED_BASE_URL) -> VocabEmbeddings:
    """Build and persist the vocab-embedding cache."""
    from inferdpt.tokeniser import Tokeniser

    vocab = Tokeniser().english_vocab(limit=limit)  # cl100k V (tokeniser_path is legacy/unused)
    # Served embedding models (qwen3-embedding-0.6b, jina) already return unit-norm
    # vectors — like OpenAI ada-002 in the reference — so no re-normalization here.
    # In-process sources with raw norms (build_from_model_matrix / _static_vectors /
    # _sentence_model) normalize in their own handler. Contract: every cached φ is unit-norm.
    matrix = embed(vocab, model=model, base_url=base_url)
    ve = VocabEmbeddings(vocab, matrix)
    ve.save(out)
    return ve


def build_from_model_matrix(words: list[str], model_id: str, out: str | Path,
                            *, normalize: bool = True) -> VocabEmbeddings:
    """Build φ from an LLM's input-embedding matrix (tier-1), mean-pooling a word's
    subword token rows. Runs offline from the HF cache (torch+transformers on the host
    `.venv` GPU). Keeps the given `words` order for a fair A/B against the API cache."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    emb = model.get_input_embeddings().weight.detach().cpu().numpy()  # [vocab, hid]

    vecs, kept = [], []
    for w in words:
        ids = tok(" " + w, add_special_tokens=False).input_ids  # leading space = word-start
        if not ids:
            continue
        vecs.append(emb[ids].mean(0))
        kept.append(w)
    M = np.asarray(vecs, dtype=np.float32)
    if normalize:
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
    ve = VocabEmbeddings(kept, M)
    ve.save(out)
    return ve


def build_from_sentence_model(words: list[str], model_id: str, out: str | Path,
                              *, normalize: bool = True, batch: int = 256) -> VocabEmbeddings:
    """Build φ from a sentence-transformers model (e.g. Phrase-BERT). Embeds each token
    string directly; row-aligned to `words`. CPU-runnable for ~110M models."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device="cpu")
    M = model.encode(words, batch_size=batch, normalize_embeddings=normalize,
                     show_progress_bar=False).astype(np.float32)
    ve = VocabEmbeddings(list(words), M)
    ve.save(out)
    return ve


def load_static_vectors(path: str | Path) -> dict[str, np.ndarray]:
    """Read a word2vec-style text file (`word v1 v2 …` per line, no header) into a
    {word: vector} dict. Covers counter-fitted / GloVe / paragram .txt."""
    vecs: dict[str, np.ndarray] = {}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            if len(parts) < 3:  # skip a possible "n dim" header line
                continue
            vecs[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return vecs


def build_from_static_vectors(words: list[str], path: str | Path, out: str | Path,
                              *, normalize: bool = True) -> VocabEmbeddings:
    """Build φ from static word vectors, keeping only `words` present in the file (in the
    given order → row-aligned for a fair A/B). Words absent from the file are dropped;
    the caller decides OOV handling (here: skip-perturb sub-vocab). CPU, offline."""
    table = load_static_vectors(path)
    kept = [w for w in words if w in table]
    M = np.stack([table[w] for w in kept]).astype(np.float32)
    if normalize:
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
    ve = VocabEmbeddings(kept, M)
    ve.save(out)
    return ve


def whiten(M: np.ndarray, k: int = 1, *, normalize: bool = True) -> np.ndarray:
    """All-but-the-top: mean-center and remove the top-k principal directions
    (the anisotropy fix). Renormalises rows so distances stay comparable."""
    c = M - M.mean(0)
    cov = (c.T @ c) / len(c)
    _, vecs = np.linalg.eigh(cov)
    top = vecs[:, -k:]  # largest-eigenvalue directions
    c = c - (c @ top) @ top.T
    if normalize:
        c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)
    return c.astype(np.float32)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the RANTEXT vocab-embedding cache.")
    ap.add_argument("--out", default="data/vocab", help="output path stem (.json/.npy)")
    ap.add_argument("--limit", type=int, default=12000)
    ap.add_argument("--source", choices=["api", "matrix"], default="api")
    ap.add_argument("--from-vocab", help="reuse the word list from this cache stem (matrix mode)")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B", help="HF model id for matrix mode")
    ap.add_argument("--whiten", type=int, default=0, help="remove top-k PCs after building")
    args = ap.parse_args()

    if args.source == "matrix":
        words = json.loads(Path(args.from_vocab).with_suffix(".json").read_text())
        ve = build_from_model_matrix(words, args.model, args.out)
    else:
        ve = build(args.out, limit=args.limit)
    if args.whiten:
        ve = VocabEmbeddings(ve.vocab, whiten(ve.matrix, args.whiten))
        ve.save(args.out)
    d = ve.sensitivity
    print(f"built {len(ve.vocab)} tokens, dim {ve.matrix.shape[1]}, "
          f"Δφ[mean={d.mean():.4f} max={d.max():.4f}]")
    print(f"saved to {args.out}.json / .npy")
