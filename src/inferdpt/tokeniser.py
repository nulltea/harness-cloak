"""Tokeniser — cl100k (tiktoken) BPE tokenizer + the RANTEXT perturbation vocab V.

Matches the InferDPT reference (`func.py`): the document splitter and V live in the
**same cl100k token space**, so each surface token is looked up directly in V. There is
no whole-word/dictionary filter — that filter is what dropped names and rare words
entirely (a token split like ``sarah → ['sar','ah']`` must hit V, not vanish).

One leading space is stripped from every token to match the reference's
``if origin_token[0]==' ': origin_token=origin_token[1:]`` normalisation and the keys
already stored in ``data/vocab.json`` (the canonical 12k cl100k V).
"""

from __future__ import annotations

import tiktoken

ENCODING = "cl100k_base"


class Tokeniser:
    def __init__(self, encoding: str = ENCODING) -> None:
        self.enc = tiktoken.get_encoding(encoding)

    def _piece(self, tid: int) -> str:
        """Decode one token id to its surface string, stripping a single leading space."""
        s = self.enc.decode_single_token_bytes(tid).decode("latin-1")
        return s[1:] if s[:1] == " " else s

    def surfaces(self, text: str) -> list[str]:
        """cl100k token surface strings for `text` (one leading space stripped)."""
        return [self._piece(t) for t in self.enc.encode(text)]

    def english_vocab(self, limit: int = 12000) -> list[str]:
        """V = the first `limit` cl100k tokens that contain a letter/digit
        (space-stripped, deduped, id order ≈ frequency). Sub-word pieces are kept on
        purpose: it is what lets names tokenise into in-V pieces and be perturbed rather
        than dropped. Reproduces ``data/vocab.json``."""
        seen: set[str] = set()
        vocab: list[str] = []
        for tid in range(self.enc.n_vocab):
            p = self._piece(tid)
            if not p or p in seen or not any(c.isalpha() for c in p):
                continue
            seen.add(p)
            vocab.append(p)
            if len(vocab) >= limit:
                break
        return vocab
