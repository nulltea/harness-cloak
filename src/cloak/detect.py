"""PII/QI span detection: GLiNER (zero-shot, TAB categories) ∪ Presidio (patterns).

Plan: docs/plans/2026-07-02-d1-prototype-implementation.md.
"""
import re
from dataclasses import dataclass

# Zero-shot label phrase -> TAB entity_type. Phrasing matters for GLiNER; tune only here.
GLINER_LABELS = {
    "person name": "PERSON",
    "organization, company, court or institution": "ORG",
    "location, address, city or country": "LOC",
    "date, time or duration": "DATETIME",
    "case number, reference number or identification code": "CODE",
    "quantity, amount of money or percentage": "QUANTITY",
    "nationality, ethnicity, religion, profession or age": "DEM",
    "other identifying attribute or event": "MISC",
}

# Presidio entity -> TAB entity_type (only types its default recognizers emit).
PRESIDIO_MAP = {
    "PERSON": "PERSON", "LOCATION": "LOC", "NRP": "DEM", "DATE_TIME": "DATETIME",
    "EMAIL_ADDRESS": "CODE", "PHONE_NUMBER": "CODE", "IBAN_CODE": "CODE",
    "CREDIT_CARD": "CODE", "US_SSN": "CODE", "IP_ADDRESS": "CODE",
    "MEDICAL_LICENSE": "CODE", "US_DRIVER_LICENSE": "CODE", "US_PASSPORT": "CODE",
    "REF_CODE": "CODE", "MONEY": "QUANTITY",
    # URL deliberately unmapped: Reddit ellipses ("here...co") false-positive as .co domains
}

_PRONOUNS = {"i", "me", "my", "mine", "you", "your", "he", "him", "his", "she", "her",
             "it", "its", "we", "us", "our", "they", "them", "their", "rn", "ngl"}


@dataclass
class Span:
    start: int
    end: int
    text: str
    type: str      # TAB entity_type
    score: float
    source: str    # "gliner" | "presidio"
    chain: int = -1  # coref chain id (set by coref_chains), -1 = unclustered


def _chunks(text: str, max_chars: int = 1200):
    """Split on line/sentence boundaries into ~max_chars windows; yield (offset, chunk)."""
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        if end < len(text):
            cut = max(text.rfind("\n", pos, end), text.rfind(". ", pos, end))
            if cut > pos + max_chars // 2:
                end = cut + 1
        yield pos, text[pos:end]
        pos = end


def _guarded_map_entities_to_original(self, outputs, valid_to_orig_idx,
                                      all_start_token_idx_to_text_idx,
                                      all_end_token_idx_to_text_idx, valid_texts, num_original_texts):
    """Drop-in for gliner BaseEncoderGLiNER._map_entities_to_original with a bounds guard.

    Some fine-tuned span models (observed on the deberta-v3-large PII fine-tune at threshold < ~0.1)
    emit low-confidence spans whose token indices land in the PADDING region, past the real sequence
    (e.g. start=225 into a 203-token map). Upstream indexes the token->char map unguarded and raises
    IndexError. Those spans map to no real text, so we drop them — this only fires on phantom padding
    predictions (a no-op for models that don't produce them, e.g. the base fine-tune). NOT a threshold
    change: the operating point is untouched; only out-of-range predictions are discarded.
    """
    all_entities = [[] for _ in range(num_original_texts)]
    for valid_i, output in enumerate(outputs):
        smap = all_start_token_idx_to_text_idx[valid_i]
        emap = all_end_token_idx_to_text_idx[valid_i]
        entities = []
        for span in output:
            if span.start >= len(smap) or span.end >= len(emap):
                continue                                   # phantom span in the padding region
            s, e = smap[span.start], emap[span.end]
            ent = {"start": s, "end": e, "text": valid_texts[valid_i][s:e],
                   "label": span.entity_type, "score": span.score}
            if span.class_probs is not None:
                ent["class_probs"] = span.class_probs
            entities.append(ent)
        all_entities[valid_to_orig_idx[valid_i]] = entities
    return all_entities


def _install_gliner_bounds_guard():
    """Idempotently patch the gliner span->text mapping with the bounds-guarded version above."""
    from gliner.model import BaseEncoderGLiNER
    if getattr(BaseEncoderGLiNER._map_entities_to_original, "_bounds_guarded", False):
        return
    _guarded_map_entities_to_original._bounds_guarded = True
    BaseEncoderGLiNER._map_entities_to_original = _guarded_map_entities_to_original


class Detector:
    # Deployment default (decided 2026-07-04): the multi-domain fine-tune v2 — TAB QUASI 0.979,
    # generality 0.872; research-wiki/training/2026-07-04-ft-detector-quasi.md. Threshold 0.3 =
    # the record's cross-domain operating point (TAB's own op point is 0.02, corpus-specific).
    # Stock fallback: gliner_model="urchade/gliner_small-v2.1".
    def __init__(self, gliner_model: str = "data/models/pii_gliner_multidomain/checkpoint-2479",
                 threshold: float = 0.3, batch_size: int = 16):
        import torch
        from gliner import GLiNER
        from presidio_analyzer import AnalyzerEngine
        _install_gliner_bounds_guard()   # guard against padding-region phantom spans (see function docstring)
        self.threshold = threshold
        self.batch_size = batch_size
        self.gliner = GLiNER.from_pretrained(gliner_model)
        if torch.cuda.is_available():
            self.gliner = self.gliner.to("cuda")
        self.presidio = AnalyzerEngine()
        from presidio_analyzer import Pattern, PatternRecognizer
        self.presidio.registry.add_recognizer(PatternRecognizer(
            supported_entity="REF_CODE", name="numeric_reference",
            patterns=[Pattern("num-slash-num", r"\b\d{3,6}/\d{2,4}\b", 0.6)]))
        _numword = (r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
                    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
                    r"million|billion|and|a)")
        self.presidio.registry.add_recognizer(PatternRecognizer(
            supported_entity="MONEY", name="money_amount",
            patterns=[Pattern("amount-currency",
                              r"(?:[$€£]\s?[\d,]+(?:\.\d+)?[kKmM]?|\b[\d,]+(?:\.\d+)?[kKmM]?\s?"
                              r"(?:dollars?|euros?|pounds?|USD|EUR|GBP|NOK|kr)\b)", 0.6),
                      Pattern("bare-k-amount", r"\b\d{1,4}(?:\.\d+)?[kKmM]\b", 0.4),
                      Pattern("spelled-amount",
                              rf"(?i)\b(?:{_numword}[\s-]+){{1,6}}(?:dollars?|euros?|pounds?)\b", 0.6)]))
        self.labels = list(GLINER_LABELS)

    def detect(self, text: str) -> list[Span]:
        spans = []
        offsets, texts = zip(*_chunks(text)) if text.strip() else ((), ())
        for off, ents in zip(offsets, self.gliner.batch_predict_entities(
                list(texts), self.labels, threshold=self.threshold, batch_size=self.batch_size)):
            spans += [Span(off + e["start"], off + e["end"], e["text"],
                           GLINER_LABELS[e["label"]], e["score"], "gliner") for e in ents]
        for r in self.presidio.analyze(text=text, language="en"):
            if r.entity_type in PRESIDIO_MAP:
                spans.append(Span(r.start, r.end, text[r.start:r.end],
                                  PRESIDIO_MAP[r.entity_type], r.score, "presidio"))
        spans = [s for s in spans  # pure symbol/emoji spans or bare pronouns: never identifiers
                 if re.search(r"[A-Za-z0-9]", s.text) and s.text.lower() not in _PRONOUNS]
        return _dedupe(spans)


def _dedupe(spans: list[Span]) -> list[Span]:
    """Overlapping spans: keep the widest, then highest score."""
    out = []
    for s in sorted(spans, key=lambda s: (s.start, -(s.end - s.start), -s.score)):
        if not any(s.start < o.end and o.start < s.end for o in out):
            out.append(s)
    return out


def coref_chains(text: str, spans: list[Span]) -> list[Span]:
    """Attach chain ids by surface aliasing: same-type spans whose casefolded token sets
    overlap (or one contains the other) share a chain.

    ponytail: string-alias coref — fastcoref 2.1.6 is incompatible with transformers 5.12
    (FCorefModel hits removed modeling internals). Aliasing covers placeholder consistency
    across name variants; upgrade to a real coref model for the TAB pass, where nominal
    anaphora ("the applicant") matters.
    """
    chains: list[tuple[str, set]] = []  # (type, token set)
    for s in sorted(spans, key=lambda s: s.start):
        toks = {t for t in s.text.lower().split() if len(t) > 2}
        s.chain = -1
        for ci, (ctype, ctoks) in enumerate(chains):
            if ctype == s.type and toks and (toks & ctoks):
                s.chain = ci
                ctoks |= toks
                break
        if s.chain < 0:
            chains.append((s.type, toks))
            s.chain = len(chains) - 1
    return spans


if __name__ == "__main__":  # offline-ish self-check (downloads models on first run)
    det = Detector()
    text = ("Sarah Johnson, a 34-year-old cardiologist at Novo Nordisk in Oslo, "
            "was diagnosed on March 3, 2021, case ref 36110/97. Contact: sarah.j@nn.dk.")
    got = det.detect(text)
    for s in got:
        print(f"{s.score:.2f} {s.source:8s} {s.type:8s} {s.text!r}")
    types = {s.type for s in got}
    assert {"PERSON", "ORG", "LOC", "DATETIME", "CODE"} <= types, types
    assert all(text[s.start:s.end] == s.text for s in got if s.source == "gliner")
    print("detect.py self-check OK")
