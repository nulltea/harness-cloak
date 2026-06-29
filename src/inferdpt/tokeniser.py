"""Tokeniser — wraps a Gemma tokenizer and builds the English perturbation vocab.

The tokeniser does two jobs for RANTEXT:
1. split a document into token surface strings (`surfaces`), and
2. define the vocabulary V of candidate replacement tokens (`english_vocab`).

Gemma's tokenizer is loaded offline from the HF cache (ungated `unsloth/gemma-2-2b`
mirror), so no auth/network is needed. Gemma 2 / 3n share the same SentencePiece
vocab; the perturbed text is re-tokenised by the remote model anyway, so an exact
match to the served E4B is not required.
"""

from __future__ import annotations

import glob
import os

from tokenizers import Tokenizer

SPACE = "▁"  # SentencePiece word-boundary marker (▁)
_CACHE_GLOB = "~/.cache/huggingface/hub/models--*gemma-2*/snapshots/*/tokenizer.json"
_WORDLIST = "/usr/share/dict/words"


def _resolve(path: str | None) -> str:
    if path:
        return path
    hits = glob.glob(os.path.expanduser(_CACHE_GLOB))
    if not hits:
        raise FileNotFoundError(
            "No gemma tokenizer.json in HF cache; pass path= explicitly."
        )
    return hits[0]


def _english_words() -> set[str]:
    with open(_WORDLIST, encoding="latin-1") as f:
        return {w.strip().lower() for w in f if w.strip().isalpha()}


class Tokeniser:
    def __init__(self, path: str | None = None) -> None:
        self.tok = Tokenizer.from_file(_resolve(path))

    def surfaces(self, text: str) -> list[str]:
        """Token surface strings for `text` (▁ stripped; special tokens dropped)."""
        out: list[str] = []
        for i in self.tok.encode(text).ids:
            piece = self.tok.id_to_token(i)
            if piece is None or (piece.startswith("<") and piece.endswith(">")):
                continue  # skip <bos>/<eos>/etc.
            out.append(piece.replace(SPACE, ""))
        return out

    def english_vocab(self, limit: int = 12000, min_len: int = 2) -> list[str]:
        """Word-start (▁) tokens that are English dictionary words.

        Returned in Gemma id order (≈ frequency), truncated to `limit` — so the
        vocab is the ~`limit` most common English words.
        """
        words = _english_words()
        cand: list[tuple[int, str]] = []
        for piece, i in self.tok.get_vocab().items():
            if not piece.startswith(SPACE):
                continue
            w = piece[len(SPACE):]
            if len(w) < min_len or not (w.isascii() and w.isalpha()):
                continue
            if w.lower() in words:
                cand.append((i, w))
        cand.sort()  # by id ≈ frequency
        seen: set[str] = set()
        vocab: list[str] = []
        for _, w in cand:
            if w not in seen:
                seen.add(w)
                vocab.append(w)
            if len(vocab) >= limit:
                break
        return vocab
