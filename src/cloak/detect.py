"""PII/QI span detection: GLiNER (zero-shot, TAB categories) ∪ Presidio (patterns).

Plan: docs/plans/2026-07-02-d1-prototype-implementation.md P0.3/P1.
"""
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
    "CREDIT_CARD": "CODE", "US_SSN": "CODE", "URL": "CODE", "IP_ADDRESS": "CODE",
    "MEDICAL_LICENSE": "CODE", "US_DRIVER_LICENSE": "CODE", "US_PASSPORT": "CODE",
    "REF_CODE": "CODE",
}


@dataclass
class Span:
    start: int
    end: int
    text: str
    type: str      # TAB entity_type
    score: float
    source: str    # "gliner" | "presidio"


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


class Detector:
    def __init__(self, gliner_model: str = "urchade/gliner_small-v2.1",
                 threshold: float = 0.3, batch_size: int = 16):
        import torch
        from gliner import GLiNER
        from presidio_analyzer import AnalyzerEngine
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
        return _dedupe(spans)


def _dedupe(spans: list[Span]) -> list[Span]:
    """Overlapping spans: keep the widest, then highest score."""
    out = []
    for s in sorted(spans, key=lambda s: (s.start, -(s.end - s.start), -s.score)):
        if not any(s.start < o.end and o.start < s.end for o in out):
            out.append(s)
    return out


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
