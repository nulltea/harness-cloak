"""Retrieve-then-verify profile matcher: exact fast path, else embedding retrieval + NLI.

Spec: docs/specs/substitutor-profile-match-retrieve-verify.md. Wired in as the substitutor's
batched pre-pass (match_spans_batch -> lattice_for(proposal=...)); match_profile_entry remains
the single-span entry point for other callers.
"""
import hashlib
import json
import logging
import math
from collections import defaultdict
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
# One globally pinned operating point for semantic profile certification.  This
# is deliberately shared across runtime types and callers: it is a membership
# test, not a per-type utility/privacy calibration knob.
ENTRY_ROOT_MARGIN = 0.15
# A level must be materially below this runtime type's broadest anonymity
# class to establish semantic-entry specificity.  This is one global pinned
# operating point, never a per-type/model/privacy calibration.
ANONYMITY_DISCRIMINATIVE_FRACTION = 0.10
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


@dataclass(frozen=True)
class _TypeCertificationStats:
    root_level: str
    discriminative_levels_by_entry: dict[str, frozenset[str]]


@lru_cache(maxsize=16)
def _profile_certification_stats(profiles_path: str,
                                 runtime_type: str) -> _TypeCertificationStats | None:
    """Derive a type root and profile-discriminative levels from the artifact.

    Every level must carry a finite, positive ``level_counts`` value.  Missing
    or malformed statistics make semantic certification unavailable for the
    type rather than inviting an ungrounded fallback.
    """
    try:
        artifact = lp.load_profiles(Path(profiles_path))
        entries = artifact.get("profiles", {}).get(runtime_type)
        if not isinstance(entries, dict) or not entries:
            return None
        memberships: dict[str, set[str]] = defaultdict(set)
        display: dict[str, str] = {}
        depth_sum: dict[str, float] = defaultdict(float)
        count_sum: dict[str, float] = defaultdict(float)
        entry_levels: dict[str, list[tuple[str, float]]] = {}
        type_max_anonymity = 0.0
        for canonical, row in entries.items():
            if not isinstance(canonical, str) or not isinstance(row, dict):
                return None
            levels = row.get("levels")
            level_counts = row.get("level_counts")
            if (not isinstance(levels, list) or not levels or
                    not isinstance(level_counts, dict)):
                return None
            normalized_levels: list[tuple[str, float]] = []
            seen: set[str] = set()
            for index, level in enumerate(levels):
                if not isinstance(level, str) or not level.strip():
                    return None
                normalized = lp._norm(level)
                if not normalized or normalized in seen or level not in level_counts:
                    return None
                try:
                    level_count = float(level_counts[level])
                except (TypeError, ValueError):
                    return None
                if not math.isfinite(level_count) or level_count <= 0.0:
                    return None
                seen.add(normalized)
                normalized_levels.append((normalized, level_count))
                type_max_anonymity = max(type_max_anonymity, level_count)
                memberships[normalized].add(canonical)
                display.setdefault(normalized, level.strip())
                depth = index / max(1, len(levels) - 1)
                depth_sum[normalized] += depth
                count_sum[normalized] += math.log(level_count)
            entry_levels[canonical] = normalized_levels
        if not memberships or not math.isfinite(type_max_anonymity) or type_max_anonymity <= 0.0:
            return None
        # The type root is the most widely shared, then shallowest, level.
        # Counts are only a deterministic final structural tie-break.
        root_key = min(
            memberships,
            key=lambda key: (-len(memberships[key]),
                             -(depth_sum[key] / len(memberships[key])),
                             -(count_sum[key] / len(memberships[key])), key),
        )
        n_entries = len(entries)
        root_members = memberships[root_key]
        # A root can be absent from some entries in an incomplete artifact.
        # Equal membership still makes a level equally broad, so neither it nor
        # a root synonym may provide semantic-entry specificity evidence.
        root_equivalent = {
            level for level, members in memberships.items()
            if members == root_members
        }
        anonymity_limit = type_max_anonymity * ANONYMITY_DISCRIMINATIVE_FRACTION
        if not math.isfinite(anonymity_limit) or anonymity_limit <= 0.0:
            return None
        discriminative = {
            canonical: frozenset(
                level for level, level_count in levels
                if (level_count < anonymity_limit and
                    len(memberships[level]) < n_entries and
                    level not in root_equivalent)
            )
            for canonical, levels in entry_levels.items()
        }
        return _TypeCertificationStats(display[root_key], discriminative)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


@dataclass
class MatchResult:
    levels: list[str]
    kind: str            # "exact" | "semantic"
    deterministic: bool
    similarity: float
    entry: str | None    # matched canonical (exact and semantic hits)
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
# _PROPOSAL_CACHE shares the convention: after regenerating profiles in-process, call
# load_embindex.cache_clear() and clear _PROPOSAL_CACHE.
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
                      nli_batch_fn=None, entry_certify_batch_fn=None,
                      entry_reverse_entailment_batch_fn=None) -> dict[tuple[str, str], "MatchResult | None"]:
    """Document-level pre-pass: one embed batch for uncached misses, wave-batched NLI.
    Returns an entry for every submitted span_key; None = abstain (fail closed)."""
    profiles_path = Path(profiles_path or lp.DEFAULT_PROFILE_PATH)
    index_path = Path(index_path) if index_path else _index_path_for(profiles_path)

    # first-context-wins mirrors the substitutor's per-surface reuse invariant (by_surface):
    # one certification per unique surface, in the first occurrence's sentence; repeats inherit.
    todo: dict[tuple[str, str], tuple[str, str]] = {}   # key -> (span_text, context); first wins
    for span_text, runtime_type, context in items:
        todo.setdefault(span_key(span_text, runtime_type), (span_text, context))

    out: dict[tuple[str, str], MatchResult | None] = {}
    misses: list[tuple[tuple[str, str], str, str]] = []  # (key, span_text, context)
    for key, (span_text, context) in todo.items():
        got = lp.lookup_entry(span_text, key[0], profiles_path)
        if got:
            out[key] = MatchResult(list(got[1]), "exact", True, 1.0, got[0])
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

    # cap-clear BEFORE computing uncached, so any of this batch's keys wiped here get
    # re-embedded in the same single call (never left dangling for the unresolved lookup).
    if len(_PROPOSAL_CACHE) > _PROPOSAL_CACHE_MAX:
        _PROPOSAL_CACHE.clear()

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
        try:
            for (k, _), q in zip(uncached, vecs):
                _PROPOSAL_CACHE[(str(index_path), k[0], k[1])] = _retrieve(index, k[0], q)
        except Exception:
            _warn_exact_only(str(index_path), "embedding output failed retrieval")
            return out

    if nli_batch_fn is None:
        from cloak.lattice import nli_gate_batch
        nli_batch_fn = lambda jobs: nli_gate_batch(jobs, thresh=NLI_THRESH)
    if entry_certify_batch_fn is None:
        from cloak.lattice import nli_entry_certify_batch
        entry_certify_batch_fn = nli_entry_certify_batch
    if entry_reverse_entailment_batch_fn is None:
        from cloak.lattice import nli_entry_reverse_entailment_batch
        entry_reverse_entailment_batch_fn = nli_entry_reverse_entailment_batch

    # wave-batched best-first certification: wave w tries every unresolved key's w-th candidate
    unresolved = [(k, s, c, _PROPOSAL_CACHE[(str(index_path), k[0], k[1])])
                  for k, s, c in misses]
    for wave in range(TOP_K):
        jobs, owners = [], []
        for k, s, c, cands in unresolved:
            if wave < len(cands):
                canonical, sim = cands[wave]
                stats = _profile_certification_stats(str(profiles_path), k[0])
                levels = lp.lookup_levels(canonical, k[0], profiles_path)
                if stats is None or not levels or not stats.discriminative_levels_by_entry.get(canonical):
                    continue
                jobs.append((s, c, levels))
                owners.append((k, s, c, canonical, sim, stats))
        if not jobs:
            break
        try:
            results = nli_batch_fn(jobs)
        except Exception:  # certifier failure degrades like embed failure: abstain, never raise
            _warn_exact_only(str(index_path), "nli certifier failed")
            return out
        if not isinstance(results, (list, tuple)) or len(results) != len(owners):
            _warn_exact_only(str(index_path), "nli certifier returned malformed results")
            return out
        entry_jobs, accepted = [], []
        for owner, approved in zip(owners, results):
            k, surface, context, canonical, sim, stats = owner
            if not isinstance(approved, (list, tuple)):
                _warn_exact_only(str(index_path), "nli certifier returned malformed results")
                return out
            valid_approved: list[tuple[str, float | None]] = []
            candidate_levels = set(lp.lookup_levels(canonical, k[0], profiles_path) or [])
            for item in approved:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    _warn_exact_only(str(index_path), "nli certifier returned malformed results")
                    return out
                level, score = item
                if not isinstance(level, str) or level not in candidate_levels:
                    _warn_exact_only(str(index_path), "nli certifier returned malformed results")
                    return out
                if score is not None:
                    try:
                        score = float(score)
                    except (TypeError, ValueError):
                        _warn_exact_only(str(index_path), "nli certifier returned malformed results")
                        return out
                    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                        _warn_exact_only(str(index_path), "nli certifier returned malformed results")
                        return out
                valid_approved.append((level, score))
            discriminative = stats.discriminative_levels_by_entry[canonical]
            if valid_approved and any(lp._norm(level) in discriminative for level, _ in valid_approved):
                entry_jobs.append((surface, context, canonical, stats.root_level))
                accepted.append((owner, valid_approved))
        if not entry_jobs:
            continue
        try:
            entry_scores = entry_certify_batch_fn(entry_jobs)
        except Exception:
            _warn_exact_only(str(index_path), "entry-membership certifier failed")
            return out
        if not isinstance(entry_scores, (list, tuple)) or len(entry_scores) != len(accepted):
            _warn_exact_only(str(index_path), "entry-membership certifier returned malformed scores")
            return out
        reverse_jobs = [(owner[1], owner[2], owner[3]) for owner, _ in accepted]
        try:
            reverse_scores = entry_reverse_entailment_batch_fn(reverse_jobs)
        except Exception:
            _warn_exact_only(str(index_path), "reverse-entailment certifier failed")
            return out
        if not isinstance(reverse_scores, (list, tuple)) or len(reverse_scores) != len(accepted):
            _warn_exact_only(str(index_path), "reverse-entailment certifier returned malformed scores")
            return out
        resolved = set()
        for (owner, approved), scores, reverse_score in zip(accepted, entry_scores, reverse_scores):
            if (not isinstance(scores, (list, tuple)) or len(scores) != 2):
                _warn_exact_only(str(index_path), "entry-membership certifier returned malformed scores")
                return out
            try:
                entry_score, root_score = (float(scores[0]), float(scores[1]))
            except (TypeError, ValueError):
                _warn_exact_only(str(index_path), "entry-membership certifier returned malformed scores")
                return out
            if (not math.isfinite(entry_score) or not math.isfinite(root_score) or
                    not 0.0 <= entry_score <= 1.0 or not 0.0 <= root_score <= 1.0):
                _warn_exact_only(str(index_path), "entry-membership certifier returned malformed scores")
                return out
            try:
                reverse_score = float(reverse_score)
            except (TypeError, ValueError):
                _warn_exact_only(str(index_path), "reverse-entailment certifier returned malformed scores")
                return out
            if not math.isfinite(reverse_score) or not 0.0 <= reverse_score <= 1.0:
                _warn_exact_only(str(index_path), "reverse-entailment certifier returned malformed scores")
                return out
            if entry_score < NLI_THRESH or entry_score - root_score < ENTRY_ROOT_MARGIN:
                continue
            if reverse_score >= NLI_THRESH:
                continue
            k, _, _, canonical, sim, _ = owner
            nli = None if any(score is None for _, score in approved) else max(score for _, score in approved)
            out[k] = MatchResult([level for level, _ in approved], "semantic", False, sim, canonical, nli=nli)
            resolved.add(k)
        unresolved = [u for u in unresolved if u[0] not in resolved]
        if not unresolved:
            break
    return out


def match_profile_entry(span_text, runtime_type, context, *, profiles_path=None,
                        index_path=None, embed_fn=None, nli_fn=None,
                        entry_certify_batch_fn=None,
                        entry_reverse_entailment_batch_fn=None) -> MatchResult | None:
    """Thin single-span wrapper over match_spans_batch, preserving the nli_fn contract."""
    nli_batch_fn = None
    if nli_fn is not None:  # adapt list-returning single-job fn; scores unavailable -> None
        nli_batch_fn = lambda jobs: [[(c, None) for c in nli_fn(e, ctx, cands)]
                                     for e, ctx, cands in jobs]
    got = match_spans_batch([(span_text, runtime_type, context)],
                            profiles_path=profiles_path, index_path=index_path,
                            embed_fn=embed_fn, nli_batch_fn=nli_batch_fn,
                            entry_certify_batch_fn=entry_certify_batch_fn,
                            entry_reverse_entailment_batch_fn=entry_reverse_entailment_batch_fn)
    return got[span_key(span_text, runtime_type)]
