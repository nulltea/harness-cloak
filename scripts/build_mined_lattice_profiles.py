"""Mine lattice profile rows from clinical training samples.

The miner runs GLiNER over clinical task documents with a small domain label set, skips spans already
covered by the common profile artifact, copies matching rows from the fine artifact, and emits conservative
fail-closed rows for still-uncovered spans.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

from cloak.corpora import load_task_docs
from cloak.detection.detect import strip_dose_suffix
from cloak.lattice.profiles import SCHEMA_VERSION, validate_profile_artifact
from cloak.lattice.profile_match import span_key
from cloak.detection.span_gate import gate_fingerprint, gate_spans

DETECTOR_LABELS = [
    "condition",
    "medical process",
    "drug",
    "injury",
    "organization medical facility",
    # not mapped to a runtime type -- offered so GLiNER puts verbalized doses here instead of
    # absorbing them into the drug span ("flomax zero point four milligrams" -> "flomax")
    "dosage",
]

LABEL_TO_RUNTIME_TYPE = {
    "condition": "health-condition",
    "injury": "injury",
    "medical process": "medical-procedure",
    "drug": "drug",
    "organization medical facility": "organization-medical-facility",
}

FALLBACK_LEVELS = {
    "health-condition": ["medical condition"],
    "injury": ["injury"],
    "medical-procedure": ["medical procedure"],
    "drug": ["medication"],
    "organization-medical-facility": ["healthcare organization"],
}

GENERIC_SURFACES = {
    "health-condition": {"condition", "conditions", "medical condition", "medical conditions"},
    "injury": {"injury", "injuries", "trauma", "wound", "wounds"},
    "medical-procedure": {"medical process", "medical procedure", "procedure", "procedures", "process"},
    "drug": {"drug", "drugs", "medication", "medications", "medicine", "medicines"},
    "organization-medical-facility": {
        "medical facility",
        "medical facilities",
        "healthcare organization",
        "healthcare organizations",
        "hospital",
        "clinic",
    },
}

MATCH_THRESHOLD = 0.92
FUZZY_STOP_TOKENS = {
    "and",
    "the",
    "with",
    "medical",
    "medicine",
    "medication",
    "condition",
    "conditions",
    "procedure",
    "process",
    "healthcare",
    "organization",
    "clinic",
    "hospital",
}


@dataclass(frozen=True)
class DetectedSpan:
    surface: str
    detector_label: str
    doc_id: str
    score: float


def normalize_detector_label(label: str) -> str:
    key = _norm(label)
    if key not in LABEL_TO_RUNTIME_TYPE:
        raise KeyError(f"unsupported detector label: {label}")
    return LABEL_TO_RUNTIME_TYPE[key]


def _new_stats() -> dict[str, object]:
    return {
        "detected_spans": 0,
        "unique_detected_spans": 0,
        "skipped_common": 0,
        "generic_skipped": 0,
        "noise_skipped": 0,     # gate deny-list drops (was is_noise_span)
        "gate_dropped": 0,      # gate anchor-margin drops
        "gate_retyped": 0,      # gate layer-2 retypes
        "copied_fine": 0,
        "new_entries": 0,
    }


def _gate_and_build_rows(
    unique: list[DetectedSpan],
    common_index: "ProfileIndex",
    fine_index: "ProfileIndex",
    stats: dict,
) -> dict[str, dict[str, dict]]:
    """Apply the semantic gate per unique span, then build profile rows for the survivors.

    The gate replaces the old is_noise_span call: deny-list drops keep noise_skipped, margin
    drops count gate_dropped, layer-2 retypes reassign runtime_type + count gate_retyped, keeps
    proceed unchanged. Fail-opens to keep when gate artifacts are absent."""
    prepared: list[tuple[DetectedSpan, str, str]] = []
    for span in unique:
        runtime_type = normalize_detector_label(span.detector_label)
        surface = _norm(span.surface)
        if runtime_type == "drug":
            surface = strip_dose_suffix(surface)
        if not surface:
            continue
        prepared.append((span, runtime_type, surface))

    decisions = gate_spans([(surface, rt) for _, rt, surface in prepared], "miner")
    profiles: dict[str, dict[str, dict]] = defaultdict(dict)
    for span, runtime_type, surface in prepared:
        if _is_generic_surface(runtime_type, surface):
            stats["generic_skipped"] += 1
            continue
        decision = decisions.get(span_key(surface, runtime_type))
        if decision is not None:
            if decision.action == "drop":
                stats["noise_skipped" if decision.layer == "denylist" else "gate_dropped"] += 1
                continue
            if decision.action == "retype" and decision.new_type:
                if decision.new_type not in FALLBACK_LEVELS:
                    # real entity of a type this miner doesn't build rows for (e.g. a place
                    # name detected as clinical) -> out of scope, skip rather than crash
                    stats["gate_retyped_out"] = stats.get("gate_retyped_out", 0) + 1
                    continue
                runtime_type = decision.new_type
                stats["gate_retyped"] += 1
                # re-run type-specific surface handling for the NEW type: the surface was
                # normalized/dose-stripped for its original type, not this one.
                if runtime_type == "drug":
                    surface = strip_dose_suffix(surface)
                if not surface:
                    continue
                if _is_generic_surface(runtime_type, surface):
                    stats["generic_skipped"] += 1
                    continue
        if common_index.find(runtime_type, surface):
            stats["skipped_common"] += 1
            continue
        fine_match = fine_index.find(runtime_type, surface)
        if fine_match:
            canonical, row = fine_match
            profiles[runtime_type][canonical] = copy.deepcopy(row)
            stats["copied_fine"] += 1
            continue
        row = _new_row(runtime_type, span)
        _merge_new_row(profiles[runtime_type], surface, row)
        stats["new_entries"] += 1
    return profiles


def build_rows_for_test(spans: list[DetectedSpan]) -> tuple[dict[str, dict[str, dict]], dict]:
    """Thin seam: gate + row-build over spans with no common/fine coverage (test-only)."""
    empty = ProfileIndex({"profiles": {}})
    stats = _new_stats()
    unique = _unique_spans(spans)
    stats["detected_spans"] = len(spans)
    stats["unique_detected_spans"] = len(unique)
    profiles = _gate_and_build_rows(unique, empty, empty, stats)
    stats["gate_fingerprint"] = gate_fingerprint()
    return profiles, stats


def build_mined_artifact(
    spans: list[DetectedSpan],
    common_artifact: dict,
    fine_artifact: dict,
    *,
    created: str | None = None,
) -> tuple[dict, dict[str, int]]:
    common_index = ProfileIndex(common_artifact)
    fine_index = ProfileIndex(fine_artifact)
    stats = _new_stats()

    unique = _unique_spans(spans)
    stats["detected_spans"] = len(spans)
    stats["unique_detected_spans"] = len(unique)

    profiles = _gate_and_build_rows(unique, common_index, fine_index, stats)
    stats["gate_fingerprint"] = gate_fingerprint()

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "created": created or str(date.today()),
        "sources": {
            "clinical-mined": {
                "corpus": "clinical",
                "detector_model": "knowledgator/gliner-pii-base-v1.0",
                "detector_labels": DETECTOR_LABELS,
                "common_profile": "data/lattice_profiles/comm_lattice_profiles.json",
                "fine_profile": "data/lattice_profiles/fine_lattice_profiles.json",
            }
        },
        "profiles": {runtime_type: dict(sorted(entries.items())) for runtime_type, entries in sorted(profiles.items())},
    }
    errors = validate_profile_artifact(artifact)
    if errors:
        raise ValueError("invalid mined profile artifact:\n" + "\n".join(errors[:50]))
    return artifact, stats


class ProfileIndex:
    def __init__(self, artifact: dict):
        self._exact: dict[str, dict[str, tuple[str, dict]]] = defaultdict(dict)
        self._by_token: dict[str, dict[str, list[tuple[str, str, dict]]]] = defaultdict(lambda: defaultdict(list))
        for runtime_type, entries in artifact.get("profiles", {}).items():
            for canonical, row in entries.items():
                for key in [_norm(canonical), *[_norm(alias) for alias in row.get("aliases", [])]]:
                    if not key:
                        continue
                    self._exact[runtime_type].setdefault(key, (canonical, row))
                    self._exact[runtime_type].setdefault(_singularized(key), (canonical, row))
                    for token in _index_tokens(key):
                        self._by_token[runtime_type][token].append((key, canonical, row))

    def find(self, runtime_type: str, surface: str) -> tuple[str, dict] | None:
        key = _norm(surface)
        if not key:
            return None
        for probe in (key, _singularized(key)):
            exact = self._exact.get(runtime_type, {}).get(probe)
            if exact:
                return exact
        for candidate, canonical, row in self._candidate_keys(runtime_type, key):
            if _is_fuzzy_match(key, candidate):
                return canonical, row
        return None

    def _candidate_keys(self, runtime_type: str, key: str) -> list[tuple[str, str, dict]]:
        if not key:
            return []
        key_len = len(key)
        key_tokens = max(1, len(key.split()))
        seen = set()
        out = []
        for token in _index_tokens(key):
            for candidate, canonical, row in self._by_token.get(runtime_type, {}).get(token, []):
                row_key = (candidate, canonical)
                if row_key in seen:
                    continue
                seen.add(row_key)
                cand_len = len(candidate)
                if cand_len < 4:
                    continue
                if abs(len(candidate.split()) - key_tokens) > 3 and not (
                    _whole_phrase_contains(key, candidate) or _whole_phrase_contains(candidate, key)
                ):
                    continue
                if min(key_len, cand_len) / max(key_len, cand_len) < 0.55 and not (
                    _whole_phrase_contains(key, candidate) or _whole_phrase_contains(candidate, key)
                ):
                    continue
                out.append((candidate, canonical, row))
        return out


def detect_clinical_spans(
    *,
    corpus: str = "clinical",
    limit: int | None = None,
    model: str = "knowledgator/gliner-pii-base-v1.0",
    threshold: float = 0.3,
    batch_size: int = 16,
    chunk_window_batch: int = 64,
) -> list[DetectedSpan]:
    import torch
    from gliner import GLiNER

    from cloak.detection.detect import _chunks, _encoder_max_words, _install_gliner_bounds_guard

    _install_gliner_bounds_guard()
    gliner = GLiNER.from_pretrained(model)
    max_words = _encoder_max_words(gliner)
    if torch.cuda.is_available():
        gliner = gliner.to("cuda")

    out: list[DetectedSpan] = []
    docs = load_task_docs(corpus, limit)
    chunk_rows: list[tuple[str, str]] = []
    for doc in docs:
        chunk_rows.extend((doc["id"], chunk_text) for _, chunk_text in _chunks(doc["text"], max_words=max_words))

    total = len(chunk_rows)
    for start in range(0, total, chunk_window_batch):
        batch = chunk_rows[start:start + chunk_window_batch]
        doc_ids = [row[0] for row in batch]
        texts = [row[1] for row in batch]
        for doc_id, ents in zip(doc_ids, gliner.batch_predict_entities(
            texts,
            DETECTOR_LABELS,
            threshold=threshold,
            batch_size=batch_size,
        )):
            for ent in ents:
                label = _norm(ent["label"])
                if label in LABEL_TO_RUNTIME_TYPE:
                    out.append(DetectedSpan(ent["text"], label, doc_id, float(ent.get("score", 0.0))))
        print(f"detected chunks {min(start + len(batch), total)}/{total} spans={len(out)}", flush=True)
    return out


def _unique_spans(spans: list[DetectedSpan]) -> list[DetectedSpan]:
    """Resolve each surface to a single best-scoring label. GLiNER emits the same surface under
    competing labels (e.g. "kidney stones" as both condition 0.93 and injury 0.60); keying on
    surface alone -- not (label, surface) -- picks the label the model was most confident about,
    so a surface lands in exactly one runtime profile instead of being double-counted across the
    injury/condition (and any other) type boundary."""
    best: dict[str, DetectedSpan] = {}
    for span in spans:
        surface = _norm(span.surface)
        label = _norm(span.detector_label)
        if not surface or len(surface) < 2:
            continue
        cur = best.get(surface)
        if cur is None or span.score > cur.score:
            best[surface] = DetectedSpan(surface, label, span.doc_id, span.score)
    return [best[k] for k in sorted(best)]


def _new_row(runtime_type: str, span: DetectedSpan) -> dict:
    return {
        "aliases": [],
        "levels": list(FALLBACK_LEVELS[runtime_type]),
        "source_ids": [f"mined-clinical:{span.doc_id}"],
        "count": 1.0,
    }


def _is_generic_surface(runtime_type: str, surface: str) -> bool:
    surface = _norm(surface)
    if surface in GENERIC_SURFACES.get(runtime_type, set()):
        return True
    return surface in {_norm(level) for level in FALLBACK_LEVELS.get(runtime_type, [])}


def _merge_new_row(entries: dict[str, dict], surface: str, row: dict) -> None:
    cur = entries.setdefault(surface, copy.deepcopy(row))
    cur["source_ids"] = sorted(set(cur.get("source_ids", [])) | set(row.get("source_ids", [])))
    cur["count"] = max(float(cur.get("count", 1.0)), float(row.get("count", 1.0)))


def _is_fuzzy_match(surface: str, candidate: str) -> bool:
    if surface == candidate:
        return True
    surface_s = _singularized(surface)
    candidate_s = _singularized(candidate)
    if surface_s == candidate_s:
        return True
    if _whole_phrase_contains(surface_s, candidate_s) or _whole_phrase_contains(candidate_s, surface_s):
        return True
    return SequenceMatcher(None, surface_s, candidate_s).ratio() >= MATCH_THRESHOLD


def _whole_phrase_contains(longer: str, shorter: str) -> bool:
    if len(shorter) < 4:
        return False
    return re.search(rf"(^|\s){re.escape(shorter)}($|\s)", longer) is not None


def _singularized(text: str) -> str:
    words = []
    for word in text.split():
        if len(word) > 4 and word.endswith("ies"):
            words.append(word[:-3] + "y")
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            words.append(word[:-1])
        else:
            words.append(word)
    return " ".join(words)


def _index_tokens(text: str) -> list[str]:
    tokens = []
    for token in _singularized(text).split():
        if len(token) < 4 or token in FUZZY_STOP_TOKENS:
            continue
        tokens.append(token)
    return tokens


def _norm(text: str) -> str:
    out = str(text).lower().strip()
    out = out.replace("&", " and ")
    out = re.sub(r"[^a-z0-9]+", " ", out)
    return re.sub(r"\s+", " ", out).strip()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="clinical")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="knowledgator/gliner-pii-base-v1.0")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk-window-batch", type=int, default=64)
    ap.add_argument("--common", default="data/lattice_profiles/comm_lattice_profiles.json")
    ap.add_argument("--fine", default="data/lattice_profiles/fine_lattice_profiles.json")
    ap.add_argument("--out", default="data/lattice_profiles/mined_lattice_profiles.json")
    ap.add_argument("--spans-out", default="results/mined_lattice_profile_spans.jsonl")
    args = ap.parse_args()

    spans = detect_clinical_spans(
        corpus=args.corpus,
        limit=args.limit,
        model=args.model,
        threshold=args.threshold,
        batch_size=args.batch_size,
        chunk_window_batch=args.chunk_window_batch,
    )
    artifact, stats = build_mined_artifact(spans, _read_json(Path(args.common)), _read_json(Path(args.fine)))
    _write_json(Path(args.out), artifact)
    if args.spans_out:
        path = Path(args.spans_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for span in spans:
                f.write(json.dumps(span.__dict__, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
