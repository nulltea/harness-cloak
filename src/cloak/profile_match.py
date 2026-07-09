"""Retrieve-then-verify profile matcher: exact fast path, else embedding retrieval + NLI.

Standalone MVP (docs/specs/substitutor-profile-match-retrieve-verify.md). Not wired into
cloak.lattice.lattice_for(); a later task validates it first.
"""
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from cloak import lattice_profiles as lp

TOP_K = 5
SIM_FLOOR = 0.70
NLI_THRESH = 0.6
DEFAULT_MODEL_ID = "BAAI/bge-small-en-v1.5"
SCHEMA_VERSION = 1


@dataclass
class MatchResult:
    levels: list[str]
    kind: str            # "exact" | "semantic"
    deterministic: bool
    similarity: float
    entry: str | None    # matched canonical; None for exact hits


def _index_path_for(profiles_path: Path) -> Path:
    return profiles_path.with_name(profiles_path.stem + ".embindex.npz")


def _l2norm(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


@lru_cache(maxsize=4)
def _st_model(model_id: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)


def build_embindex(profiles_path, out_path=None, model_id=DEFAULT_MODEL_ID,
                   embed_fn: Callable[[list[str]], np.ndarray] | None = None) -> Path:
    profiles_path = Path(profiles_path)
    out_path = Path(out_path) if out_path else _index_path_for(profiles_path)
    artifact = lp.load_profiles(profiles_path)

    rows, texts = [], []
    for runtime_type, entries in artifact.get("profiles", {}).items():
        for canonical, row in entries.items():
            for surface in [canonical, *row.get("aliases", [])]:
                rows.append({"runtime_type": runtime_type, "canonical": canonical,
                             "source_text": surface})
                texts.append(surface)

    if embed_fn is None:
        model = _st_model(model_id)
        embed_fn = lambda t: model.encode(t, normalize_embeddings=True)
    vectors = _l2norm(embed_fn(texts)) if texts else np.zeros((0, 0), dtype=np.float32)

    meta = {"schema_version": SCHEMA_VERSION, "model_id": model_id,
            "dim": int(vectors.shape[1]) if vectors.size else 0,
            "profile_hash": _file_sha256(profiles_path), "rows": rows}
    np.savez(out_path, vectors=vectors, meta=np.array(json.dumps(meta)))
    return out_path


class _Index:
    def __init__(self, model_id: str, vectors: np.ndarray, rows: list[dict]):
        self.model_id = model_id
        self.vectors = vectors
        self.rows = rows
        self.types = {r["runtime_type"] for r in rows}
        self._by_type: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            self._by_type.setdefault(r["runtime_type"], []).append(i)

    def type_rows(self, runtime_type: str) -> list[int]:
        return self._by_type.get(runtime_type, [])


@lru_cache(maxsize=8)
def load_embindex(index_path: str, profiles_path: str) -> _Index | None:
    path = Path(index_path)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        meta = json.loads(data["meta"].item())
        vectors = data["vectors"]
    except (OSError, ValueError, KeyError):
        return None
    if meta.get("schema_version") != SCHEMA_VERSION:
        return None
    if meta.get("profile_hash") != _file_sha256(Path(profiles_path)):
        return None
    return _Index(meta["model_id"], vectors, meta["rows"])


def match_profile_entry(span_text, runtime_type, context, *, profiles_path=None,
                        index_path=None, embed_fn=None, nli_fn=None) -> MatchResult | None:
    profiles_path = Path(profiles_path or lp.DEFAULT_PROFILE_PATH)
    index_path = Path(index_path) if index_path else _index_path_for(profiles_path)

    # 1. exact fast path
    levels = lp.lookup_levels(span_text, runtime_type, profiles_path)
    if levels:
        return MatchResult(levels, "exact", True, 1.0, None)

    # 2. degradation: missing/stale index, unknown type, or no context to certify in
    index = load_embindex(str(index_path), str(profiles_path))
    if index is None or runtime_type not in index.types or not context:
        return None

    # 3. retrieve: cosine against the type's rows, top-K above the floor
    idxs = index.type_rows(runtime_type)
    if embed_fn is None:
        model = _st_model(index.model_id)
        embed_fn = lambda t: model.encode(t, normalize_embeddings=True)
    q = _l2norm(embed_fn([span_text]))[0]
    sims = index.vectors[idxs] @ q
    kept = [(idxs[p], float(sims[p])) for p in np.argsort(-sims) if sims[p] >= SIM_FLOOR][:TOP_K]

    # dedup rows to entries (aliases share a canonical); keep max sim, order by sim desc
    best: dict[str, float] = {}
    for row_i, sim in kept:
        canonical = index.rows[row_i]["canonical"]
        if sim > best.get(canonical, -1.0):
            best[canonical] = sim
    candidates = sorted(best.items(), key=lambda kv: -kv[1])
    if not candidates:
        return None

    # 4. certify best-first
    if nli_fn is None:
        from cloak.lattice import nli_gate
        nli_fn = lambda e, c, lv: nli_gate(e, c, lv, thresh=NLI_THRESH)
    for canonical, sim in candidates:
        approved = nli_fn(span_text, context, lp.lookup_levels(canonical, runtime_type, profiles_path))
        if approved:
            return MatchResult(approved, "semantic", False, sim, canonical)
    return None
