"""Canonical vocabulary of standard abstraction-tier labels per runtime type.

Fixes paraphrase proliferation at the source: instead of letting the model reinvent a new
spelling for a shared concept every time (the reviewed drug-health-procedure run had
"pharmaceutical compound" / "pharmaceutical agent" / "pharmaceutical product" /
"pharmacological substance" all as distinct levels for the identical concept), the model is
shown a bounded slice of labels the corpus already uses for this runtime type and told to reuse
one, if any fits, rather than free-inventing new phrasing.

Seeded from:
- (drug, health-condition) the hand-curated real-world magnitude anchor tables in
  data/lattice_sources/reference/*.json (same source coherence.py's PAVA pass anchors to).
- (medical-procedure) the real ICD-10-PCS header descriptions parsed by reference_sources.py --
  no hand-curated file needed, since the source data already provides real category names.
- every level this RUN has already accepted, read from the on-disk proposed artifact (pass
  `proposed_out`). This is what actually fixes paraphrase proliferation for labels outside the
  ~40-label static anchor set -- the vast majority of what a model invents mid-run. Reading the
  live proposed artifact (rather than threading a growing vocabulary object through LangGraph
  state) works because the graph persists every accepted item before the next item's context
  packet is assembled -- items are processed strictly sequentially, never in parallel -- so
  item 500's context packet genuinely sees item 50's already-committed labels.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from cloak.lattice_producer.coherence import load_reference_anchors
from cloak.lattice_producer.reference_sources import load_icd10pcs_index


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class CanonicalVocabulary:
    def __init__(self, runtime_type: str, *, proposed_out: str | Path | None = None):
        self.runtime_type = runtime_type
        self._labels: dict[str, float] = {}
        self._run_labels: set[str] = set()
        self._seed()
        if proposed_out is not None:
            self._seed_from_run(proposed_out)

    def _seed(self) -> None:
        for label, count in load_reference_anchors(self.runtime_type).items():
            self._labels[_norm(label)] = count
        if self.runtime_type == "medical-procedure":
            for _prefix, (desc, members) in load_icd10pcs_index().items():
                label = _norm(desc)
                self._labels.setdefault(label, float(len(members)))

    def _seed_from_run(self, proposed_out: str | Path) -> None:
        path = Path(proposed_out)
        if not path.exists():
            return
        try:
            artifact: dict[str, Any] = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        entries = artifact.get("profiles", {}).get(self.runtime_type, {})
        if not isinstance(entries, dict):
            return
        for row in entries.values():
            if not isinstance(row, dict):
                continue
            level_counts = row.get("level_counts") or {}
            for level in row.get("levels", []):
                key = _norm(level)
                count = float(level_counts.get(level, 1.0))
                if key not in self._labels or key in self._run_labels:
                    self._labels[key] = count
                    self._run_labels.add(key)

    def is_from_this_run(self, label: str) -> bool:
        """True if a label came from an earlier item in this run, not a static seed -- useful
        for diagnostics/telemetry on how much the dynamic vocabulary is actually contributing."""
        return _norm(label) in self._run_labels

    def has_exact(self, label: str) -> bool:
        return _norm(label) in self._labels

    def nearest(self, candidate_label: str, k: int = 3, *, min_overlap: float = 0.0) -> list[str]:
        """Token-Jaccard nearest labels already in the vocabulary, most similar first. No
        embeddings dependency -- this only needs to catch near-duplicate paraphrases of a
        recurring tier, not deep semantic similarity. `min_overlap` filters out weak/coincidental
        single-word overlaps; callers that want a "this is basically the same label" check
        should pass a calibrated threshold rather than relying on top-k alone."""
        candidate_tokens = _tokens(candidate_label)
        if not candidate_tokens:
            return []
        scored = []
        for label in self._labels:
            if label == _norm(candidate_label):
                continue
            tokens = _tokens(label)
            if not tokens:
                continue
            overlap = len(candidate_tokens & tokens) / len(candidate_tokens | tokens)
            if overlap > min_overlap:
                scored.append((overlap, label))
        scored.sort(key=lambda pair: -pair[0])
        return [label for _, label in scored[:k]]

    def context_slice(self, n: int = 10, *, surface: str | None = None) -> list[dict]:
        """A bounded, representative slice for a context packet as {label, count} rows. When a
        surface is given, rank by token-overlap with the surface first (so the model sees the
        labels most likely to be the right reuse target), then by count; otherwise count-desc."""
        labels = list(self._labels)
        if surface:
            surface_tokens = _tokens(surface)
            def key(label):
                overlap = len(_tokens(label) & surface_tokens)
                return (-overlap, -self._labels[label])
            labels.sort(key=key)
        else:
            labels.sort(key=lambda label: -self._labels[label])
        return [{"label": label, "count": self._labels[label]} for label in labels[:n]]

    def all_labels(self) -> list[str]:
        return list(self._labels)
