"""Retrieve-then-verify profile matcher: exact fast path, else embedding retrieval + NLI.

Standalone MVP (docs/specs/substitutor-profile-match-retrieve-verify.md). Not wired into
cloak.lattice.lattice_for(); a later task validates it first.
"""
import hashlib
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from cloak import lattice_profiles as lp
from cloak.runtime_types import (COARSE_RUNTIME_TYPES, DOMAIN_RUNTIME_TYPES,
                                  FINE_DEM_TYPES, PLACEHOLDER_ONLY_TYPES)

TOP_K = 5
SIM_FLOOR = 0.70
NLI_THRESH = 0.6
DEFAULT_MODEL_ID = "BAAI/bge-small-en-v1.5"
SCHEMA_VERSION = 1

log = logging.getLogger(__name__)

# rule-based / placeholder-only / direct types never consult profiles
PROFILE_BACKED_TYPES = frozenset(
    set(COARSE_RUNTIME_TYPES) - {"DATETIME", "QUANTITY"}
    | (set(FINE_DEM_TYPES) - set(PLACEHOLDER_ONLY_TYPES) - {"age", "demographic-other"})
    | set(DOMAIN_RUNTIME_TYPES))

# proposal cache: (index_path, runtime_type, norm_surface) -> [(canonical, sim), ...]
# retrieval only — certification is context-dependent and never cached
_PROPOSAL_CACHE: dict[tuple[str, str, str], list[tuple[str, float]]] = {}
_PROPOSAL_CACHE_MAX = 100_000
_WARNED_INDEX_PATHS: set[str] = set()


@dataclass
class MatchResult:
    levels: list[str]
    kind: str            # "exact" | "semantic"
    deterministic: bool
    similarity: float
    entry: str | None    # matched canonical; None for exact hits
    nli: float | None = None  # top approved level's entailment score; None for exact / custom nli_fn


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


# Cache keyed on paths, not content: an in-process profile rewrite at the same path is not
# re-detected until cache_clear() (matches the lattice_profiles caching convention).
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


def _retrieve(index: _Index, runtime_type: str, q: np.ndarray) -> list[tuple[str, float]]:
    """Cosine retrieval against a type's rows, deduped to canonical entries (max sim)."""
    idxs = index.type_rows(runtime_type)
    if not idxs:
        return []
    sims = index.vectors[idxs] @ q
    kept = [(idxs[p], float(sims[p])) for p in np.argsort(-sims) if sims[p] >= SIM_FLOOR][:TOP_K]
    best: dict[str, float] = {}
    for row_i, sim in kept:
        canonical = index.rows[row_i]["canonical"]
        if sim > best.get(canonical, -1.0):
            best[canonical] = sim
    return sorted(best.items(), key=lambda kv: -kv[1])


def span_key(span_text: str, runtime_type: str) -> tuple[str, str]:
    return (runtime_type, lp._norm(span_text))


def _warn_exact_only(index_path: str, reason: str) -> None:
    if index_path not in _WARNED_INDEX_PATHS:
        _WARNED_INDEX_PATHS.add(index_path)
        log.warning("profile_match: %s (%s) — degrading to exact-only matching", reason, index_path)


def match_spans_batch(items, *, profiles_path=None, index_path=None, embed_fn=None,
                      nli_batch_fn=None) -> dict[tuple[str, str], "MatchResult | None"]:
    """Document-level pre-pass: one embed batch for uncached misses, wave-batched NLI.
    Returns an entry for every submitted span_key; None = abstain (fail closed)."""
    profiles_path = Path(profiles_path or lp.DEFAULT_PROFILE_PATH)
    index_path = Path(index_path) if index_path else _index_path_for(profiles_path)

    todo: dict[tuple[str, str], tuple[str, str]] = {}   # key -> (span_text, context); first wins
    for span_text, runtime_type, context in items:
        todo.setdefault(span_key(span_text, runtime_type), (span_text, context))

    out: dict[tuple[str, str], MatchResult | None] = {}
    misses: list[tuple[tuple[str, str], str, str]] = []  # (key, span_text, context)
    for key, (span_text, context) in todo.items():
        levels = lp.lookup_levels(span_text, key[0], profiles_path)
        if levels:
            out[key] = MatchResult(levels, "exact", True, 1.0, None)
        else:
            out[key] = None
            if context:
                misses.append((key, span_text, context))
    if not misses:
        return out

    index = load_embindex(str(index_path), str(profiles_path))
    if index is None:
        _warn_exact_only(str(index_path), "index missing or stale")
        return out
    misses = [(k, s, c) for k, s, c in misses if k[0] in index.types]
    if not misses:
        return out

    # one embed batch for surfaces not in the proposal cache
    uncached = [(k, s) for k, s, _ in misses
                if (str(index_path), k[0], k[1]) not in _PROPOSAL_CACHE]
    if uncached:
        try:
            if embed_fn is None:
                model = _st_model(index.model_id)
                embed_fn = lambda t: model.encode(t, normalize_embeddings=True)
            vecs = _l2norm(embed_fn([s for _, s in uncached]))
        except Exception:
            _warn_exact_only(str(index_path), "embedding model failed")
            return out
        if len(_PROPOSAL_CACHE) > _PROPOSAL_CACHE_MAX:
            _PROPOSAL_CACHE.clear()
        for (k, _), q in zip(uncached, vecs):
            _PROPOSAL_CACHE[(str(index_path), k[0], k[1])] = _retrieve(index, k[0], q)

    if nli_batch_fn is None:
        from cloak.lattice import nli_gate_batch
        nli_batch_fn = lambda jobs: nli_gate_batch(jobs, thresh=NLI_THRESH)

    # wave-batched best-first certification: wave w tries every unresolved key's w-th candidate
    unresolved = [(k, s, c, _PROPOSAL_CACHE[(str(index_path), k[0], k[1])])
                  for k, s, c in misses]
    for wave in range(TOP_K):
        jobs, owners = [], []
        for k, s, c, cands in unresolved:
            if wave < len(cands):
                canonical, sim = cands[wave]
                jobs.append((s, c, lp.lookup_levels(canonical, k[0], profiles_path)))
                owners.append((k, canonical, sim))
        if not jobs:
            break
        results = nli_batch_fn(jobs)
        resolved = set()
        for (k, canonical, sim), approved in zip(owners, results):
            if approved:
                nli = None if any(sc is None for _, sc in approved) else max(sc for _, sc in approved)
                out[k] = MatchResult([c for c, _ in approved], "semantic", False, sim, canonical, nli=nli)
                resolved.add(k)
        unresolved = [u for u in unresolved if u[0] not in resolved]
        if not unresolved:
            break
    return out


def match_profile_entry(span_text, runtime_type, context, *, profiles_path=None,
                        index_path=None, embed_fn=None, nli_fn=None) -> MatchResult | None:
    """Thin single-span wrapper over match_spans_batch, preserving the nli_fn contract."""
    nli_batch_fn = None
    if nli_fn is not None:  # adapt list-returning single-job fn; scores unavailable -> None
        nli_batch_fn = lambda jobs: [[(c, None) for c in nli_fn(e, ctx, cands)]
                                     for e, ctx, cands in jobs]
    got = match_spans_batch([(span_text, runtime_type, context)],
                            profiles_path=profiles_path, index_path=index_path,
                            embed_fn=embed_fn, nli_batch_fn=nli_batch_fn)
    return got[span_key(span_text, runtime_type)]
