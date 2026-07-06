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

# --- v7 fine-primary DEM: the detector targets these fine leaves; TAB-8's DEM is recovered by rolling
# them up (FINE_TYPE_ROLLUP) only at eval. See research-wiki/training/2026-07-05-FT-detector-v7-dem-decompose.md.
FINE_DEM_LABELS = {   # fine leaf phrase -> leaf key
    "nationality or citizenship": "nationality",
    "ethnicity or race": "ethnicity",
    "religion or religious belief": "religion",
    "profession, occupation or job title": "profession",
    "age": "age",
    "gender": "gender",
    "marital status": "marital-status",
    "health condition, disease or medical diagnosis": "health-condition",
    "sexual orientation": "sexual-orientation",
    "family role or relationship": "family-role",
    "other demographic attribute": "demographic-other",
}
FINE_DEM_PHRASE = {leaf: phrase for phrase, leaf in FINE_DEM_LABELS.items()}   # leaf key -> phrase (builder)
# inference label set under --fine-dem: the 7 non-DEM TAB phrases (as TAB types) + the fine DEM leaf phrases.
FINE_LABELS = {p: t for p, t in GLINER_LABELS.items() if t != "DEM"}
FINE_LABELS.update(FINE_DEM_LABELS)
# fine type -> TAB-8 type (every DEM leaf -> DEM; the 7 TAB types map to themselves). Used for the eval rollup.
FINE_TYPE_ROLLUP = {leaf: "DEM" for leaf in FINE_DEM_LABELS.values()}
FINE_TYPE_ROLLUP.update({t: t for t in set(GLINER_LABELS.values())})


def rollup_type(t):
    """fine leaf type -> TAB-8 type (DEM); TAB-8 types unchanged. For scoring fine predictions vs TAB gold."""
    return FINE_TYPE_ROLLUP.get(t, t)


# gazetteers/keywords for relabeling a TAB DEM span surface -> fine leaf (first-cut lexicons; unmatched ->
# demographic-other so nothing is lost; ~61% TAB-dev coverage). Shared by the builder (train relabel) and
# the gate (per-leaf gold). ponytail: expand or swap for a model-based relabeler if coverage is too low.
_NATIONALITY = {"german","austrian","polish","british","english","swedish","swiss","spanish","french",
    "american","romanian","russian","italian","dutch","greek","turkish","norwegian","danish","finnish",
    "belgian","portuguese","irish","scottish","welsh","czech","slovak","hungarian","bulgarian","ukrainian",
    "croatian","serbian","bosnian","albanian","moldovan","georgian","armenian","azerbaijani","chinese",
    "japanese","korean","indian","pakistani","afghan","iranian","iraqi","syrian","lebanese","israeli",
    "nigerian","kenyan","ghanaian","egyptian","moroccan","algerian","tunisian","ethiopian","somali","manx",
    "sierra leonean"}
_ETHNICITY = {"kurdish","gypsy","gypsies","roma","romani","sami","tamil","arab","chechen","tatar"}
_RELIGION = {"muslim","christian","catholic","protestant","jewish","hindu","buddhist","orthodox","sunni",
    "shia","islam","islamic","christianity","judaism","jehovah","evangelical","atheist","agnostic"}
_ORIENTATION = {"homosexual","homosexuality","gay","lesbian","bisexual","heterosexual","transsexual",
    "transgender","lgbt"}
_GENDER = {"male","female","man","woman","transgender man","transgender woman"}
_MARITAL = {"married","divorced","single","widow","widower","widowed","unmarried","separated"}
_FAMILY = {"father","mother","son","daughter","brother","sister","wife","husband","spouse","child",
    "children","grandmother","grandfather","grandchild","parent","sibling","cousin","uncle","aunt",
    "nephew","niece","stepson","stepdaughter","stepfather","stepmother","in-law","widow","widower"}
_CONDITION_KW = ("diabet","depress","cancer","hiv","aids","disorder","syndrome","disease","illness",
    "schizophren","tumour","tumor","psychiat","psycholog","addict","alcohol","dementia","epilep","asthma",
    "arthritis","hepatitis","paralys","mesothelioma","devitalis","korsakoff","traumatic stress","disabil",
    "blood pressure","infection","heart attack","stroke","injur","fracture","wound","amputat")
_PROFESSION = {"journalist","lawyer","doctor","nurse","teacher","engineer","judge","prosecutor","accountant",
    "officer","officers","police","policeman","soldier","professor","physician","architect","farmer",
    "driver","businessman","politician","priest","minister","author","artist","actor","scientist"}


def relabel_dem(surface):
    """TAB DEM span surface -> fine leaf key. Order matters (condition/orientation before the demonym sets,
    e.g. 'jewish' is religion not nationality). Unmatched -> demographic-other."""
    s = surface.strip().lower()
    sn = s.rstrip("s")                                              # crude singular (widows->widow)
    if any(k in s for k in _CONDITION_KW):                          return "health-condition"
    if s in _ORIENTATION or "homosexual" in s:                      return "sexual-orientation"
    if s in _RELIGION:                                              return "religion"
    if s in _ETHNICITY:                                             return "ethnicity"
    if s in _NATIONALITY or s.endswith((" national", " nationals")): return "nationality"
    if s in _GENDER:                                                return "gender"
    if s in _MARITAL or sn in _MARITAL:                             return "marital-status"
    if s in _FAMILY or sn in _FAMILY or any(w in s.split() for w in _FAMILY): return "family-role"
    if re.search(r"\b(aged|years? old)\b", s) or re.search(r"\b\d{1,3}[- ]year", s) \
       or re.fullmatch(r"\d{1,3}", s):                              return "age"
    if s in _PROFESSION or sn in _PROFESSION or s.endswith(("ist", "ologist", "ian")): return "profession"
    return "demographic-other"


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
                 threshold: float = 0.3, batch_size: int = 16, fine_dem: bool = False):
        import torch
        from gliner import GLiNER
        from presidio_analyzer import AnalyzerEngine
        _install_gliner_bounds_guard()   # guard against padding-region phantom spans (see function docstring)
        self.threshold = threshold
        self.batch_size = batch_size
        # v7: fine-primary mode prompts the fine DEM leaves; else the coarse TAB-8. self.label2type maps a
        # predicted label phrase -> its (fine or coarse) type; the gate rolls fine types up via rollup_type.
        self.fine_dem = fine_dem
        self.label2type = FINE_LABELS if fine_dem else GLINER_LABELS
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
        self.labels = list(self.label2type)

    def detect(self, text: str) -> list[Span]:
        spans = []
        offsets, texts = zip(*_chunks(text)) if text.strip() else ((), ())
        for off, ents in zip(offsets, self.gliner.batch_predict_entities(
                list(texts), self.labels, threshold=self.threshold, batch_size=self.batch_size)):
            spans += [Span(off + e["start"], off + e["end"], e["text"],
                           self.label2type[e["label"]], e["score"], "gliner") for e in ents]
        for r in self.presidio.analyze(text=text, language="en"):
            if r.entity_type in PRESIDIO_MAP:
                t = PRESIDIO_MAP[r.entity_type]
                if self.fine_dem and t == "DEM":   # fine-type Presidio's NRP span by its surface, so it
                    t = relabel_dem(text[r.start:r.end])   # doesn't clobber GLiNER fine leaves with coarse DEM
                spans.append(Span(r.start, r.end, text[r.start:r.end], t, r.score, "presidio"))
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
