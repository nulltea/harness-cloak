"""PII/QI span detection: GLiNER (zero-shot, TAB categories) ∪ Presidio (patterns).

Plan: docs/plans/2026-07-02-d1-prototype-implementation.md.
"""
import re
from dataclasses import dataclass, replace

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

_PRONOUNS = frozenset({"i", "me", "my", "mine", "you", "your", "he", "him", "his", "she",
                       "her", "it", "its", "we", "us", "our", "they", "them", "their"})
_SLANG_STOP = frozenset({"rn", "ngl"})  # chat slang; but "RN" = registered nurse in clinical text
_NOISE_FILTER_TYPES = frozenset({"drug", "health-condition", "medical-procedure", "injury"})

# Cross-type negative filter: confirmed non-entity categories for clinical drug/condition/procedure mining.
_NOISE_KEEP_PATTERNS = (
    re.compile(r"tocopherol|ascorbic|lipoic|cholecalciferol|niacin|riboflavin|thiamine|folic|"
               r"pantothen|biotin|aminobutyric|retino|calciferol|menadione|pyridoxine|cobalamin"),
    re.compile(r"\binhibitor\b"),
)
_NOISE_KEEP_TOKENS = frozenset({
    "azo", "mmr", "pcp", "cla", "pop",
    "cad", "cva", "gerd", "copd", "cf", "chf", "dvt", "uti", "tia", "ckd", "mi",
    "ms", "ra", "ed", "tb", "af", "paraldehyde",
})
_NOISE_LAB_PATTERN = re.compile(r"\b(panel|assay|titer|titre|screen|culture|antibody test|serology)\b")
_NOISE_LAB_TESTS = frozenset({
    # pt is intentionally included as prothrombin time, despite physical-therapy ambiguity.
    "cbc", "bmp", "cmp", "bnp", "psa", "hcg", "afp", "ldh", "bun", "pcr", "tsh", "esr",
    "inr", "ua", "hba1c", "a1c", "pt", "ptt", "crp", "troponin", "ferritin",
    "amylase", "lipase", "transaminase", "aminotransferase", "phosphatase",
})
_NOISE_IMAGING_DIAGNOSTICS = frozenset({
    "mri", "ct", "ct scan", "cat scan", "ecg", "ekg", "eeg", "emg", "ncs", "x ray", "xray",
    "chest x ray", "ultrasound", "echo", "echocardiogram", "angiogram", "mammogram", "dexa",
    "pet", "pet scan",
})
_NOISE_DEVICE_PATTERN = re.compile(
    r"\b(wrap|wraps|bandage|gauze|tape|swab|kit|dressing|brace|splint|sponge|applicator|"
    r"cloth|wipe|wipes|pump|pumps|catheter|catheters|stent|stents|pacemaker|pacemakers|"
    r"nebulizer|nebulizers)\b"
)
_NOISE_DEVICE_SUPPLIES = frozenset({"iud", "ace wrap"})
_NOISE_ANATOMY = frozenset({
    "arm", "leg", "knee", "hip", "shoulder", "elbow", "wrist", "ankle", "back", "lower back",
    "neck", "chest", "abdomen", "foot", "hand", "lad",
})
_NOISE_LEGAL_ADMIN_PATTERN = re.compile(
    r"\b(power of attorney|living will|driver(?:s| s)? licen[sc]e|insurance policy|employment contract|"
    r"advance directive)\b"
)
_NOISE_JUNK_NUMERIC = re.compile(r"^[\d][\d .]*$")
_NOISE_FRAGRANCE_EXCIPIENT = re.compile(r"aldehyde$|cinnamaldehyde|\blimonene\b|\blinalool\b")


# Dictation transcripts verbalize doses inside the drug span ("flomax zero point four
# milligrams", "aspirin 81 milligrams daily"); both stock and large GLiNER include them.
_DOSE_NUM_WORD = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|half|point|and|a)"
_DOSE_SUFFIX = re.compile(
    rf"\s+(?:\d+(?:\.\d+)?|{_DOSE_NUM_WORD}(?:[\s-]{_DOSE_NUM_WORD})*)"
    r"\s*(?:mg|mcg|milligrams?|micrograms?|grams?|units?)"
    r"(?:\s+(?:once|twice|three times)?\s*(?:daily|nightly|weekly|a day|per day|bid|tid|qid|qd|qhs|prn))?$"
)


def strip_dose_suffix(surface: str) -> str:
    """Strip a trailing dose (+ optional frequency) from a mined drug surface, keeping the drug."""
    return _DOSE_SUFFIX.sub("", surface).strip()


def _noise_norm(text: str) -> str:
    out = str(text).lower().strip()
    out = re.sub(r"[^a-z0-9]+", " ", out)
    return re.sub(r"\s+", " ", out).strip()


def _compact_abbreviation(surface: str) -> str:
    return surface.replace(" ", "")


def _noise_runtime_type(text: str) -> str:
    out = str(text).lower().strip().replace("_", "-")
    out = re.sub(r"\s+", "-", out)
    return out


def is_noise_span(surface: str, runtime_type: str) -> bool:
    """True iff `surface` is confirmed lab/imaging/device/anatomy/legal-admin/junk noise.

    KEEP wins first; unmatched surfaces fail open. The predicate defensively normalizes raw
    input and only applies to the queue families this filter was designed for.
    """
    runtime_type = _noise_runtime_type(runtime_type)
    if runtime_type not in _NOISE_FILTER_TYPES:
        return False
    s = _noise_norm(surface)
    if not s:
        return False
    compact = _compact_abbreviation(s)
    if compact in _NOISE_KEEP_TOKENS or any(p.search(s) or p.search(compact) for p in _NOISE_KEEP_PATTERNS):
        return False
    if _NOISE_LAB_PATTERN.search(s) or s in _NOISE_LAB_TESTS or compact in _NOISE_LAB_TESTS:
        return True
    if s in _NOISE_IMAGING_DIAGNOSTICS or compact in _NOISE_IMAGING_DIAGNOSTICS:
        return True
    if _NOISE_DEVICE_PATTERN.search(s) or s in _NOISE_DEVICE_SUPPLIES or compact in _NOISE_DEVICE_SUPPLIES:
        return True
    if s in _NOISE_ANATOMY:
        return True
    if _NOISE_LEGAL_ADMIN_PATTERN.search(s):
        return True
    if _NOISE_JUNK_NUMERIC.fullmatch(s) or _NOISE_FRAGRANCE_EXCIPIENT.search(s):
        return True
    return len(s.split()) == 1 and len(compact) <= 2


@dataclass(frozen=True)
class DetectorProfile:
    """Per-corpus detector configuration: which stop words suppress detected spans and
    whether the custom pattern recognizers (REF_CODE, MONEY) are registered. Those regexes
    are right for reddit/legal text but misfire on clinical vitals (120/80) and ranges."""
    name: str
    slang_stop_words: bool
    custom_recognizers: bool
    negative_filter: bool


PROFILES = {
    "reddit": DetectorProfile("reddit", slang_stop_words=True, custom_recognizers=True, negative_filter=False),
    "legal": DetectorProfile("legal", slang_stop_words=False, custom_recognizers=True, negative_filter=False),
    "clinical": DetectorProfile("clinical", slang_stop_words=False, custom_recognizers=False, negative_filter=True),
}


def _stop_words(profile: DetectorProfile) -> frozenset[str]:
    return _PRONOUNS | _SLANG_STOP if profile.slang_stop_words else _PRONOUNS


@dataclass
class Span:
    start: int
    end: int
    text: str
    type: str      # TAB entity_type
    score: float
    source: str    # "gliner" | "presidio" (spaCy NER) | "presidio-pattern"
    chain: int = -1  # coref chain id (set by coref_chains), -1 = unclustered


def _chunks(text: str, max_chars: int = 1200, max_words: int | None = None,
            overlap_chars: int = 200):
    """Split on line/sentence boundaries into ~max_chars windows; yield (offset, chunk).

    Never cuts mid-word: if no newline/sentence break falls in the window's second half,
    back off to the last whitespace instead of a hard character cut (a hard cut splits the
    entity under it across chunks). A fallback (non-sentence) cut can still bisect a
    MULTI-WORD entity at its internal space — the chunker cannot know "Sarah Johnson" is one
    unit — so the next window re-starts overlap_chars earlier on a word boundary: any entity
    within overlap_chars of the boundary appears whole in one chunk, and _dedupe merges the
    duplicate detections (same-type overlap keeps the widest). Sentence/newline cuts stay
    contiguous, so normally punctuated prose chunks exactly as before. max_words re-splits
    any chunk whose whitespace token count exceeds the encoder window (spaced-out OCR/ASR
    text inflates tokens ~2x per char; gliner-pii-large has max_len=768 vs 2048 base).
    """
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        sentence_cut = True
        if end < len(text):
            cut = max(text.rfind("\n", pos, end), text.rfind(". ", pos, end))
            if cut > pos + max_chars // 2:
                end = cut + 1
            else:
                sentence_cut = False
                ws = text.rfind(" ", pos + max_chars // 2, end)
                if ws > pos:
                    end = ws + 1  # word boundary, not a mid-word character cut
        chunk = text[pos:end]
        if max_words and len(chunk.split()) > max_words:
            words = list(re.finditer(r"\S+", chunk))
            for i in range(0, len(words), max_words):
                last = words[min(i + max_words, len(words)) - 1]
                yield pos + words[i].start(), chunk[words[i].start():last.end()]
        else:
            yield pos, chunk
        if sentence_cut or end >= len(text):
            pos = end
        else:
            back = text.rfind(" ", end - overlap_chars, end - 1)
            pos = back + 1 if back > pos else end  # overlap, but always strictly advance


def _encoder_max_words(gliner) -> int | None:
    """Word cap for _chunks from the model's window. gliner max_len counts words and varies
    by model (base/fine-tune 2048, gliner-pii-large 768); overflow is silently truncated by
    the encoder, so chunks must stay under it. 0.9 margin for the label prompt overhead."""
    max_len = getattr(gliner.config, "max_len", None)
    return int(max_len * 0.9) if max_len else None


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
                 threshold: float = 0.3, batch_size: int = 16, fine_dem: bool = False,
                 profile: str = "reddit"):
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
        self.max_words = _encoder_max_words(self.gliner)
        if torch.cuda.is_available():
            self.gliner = self.gliner.to("cuda")
        self.presidio = AnalyzerEngine()
        self.profile = PROFILES[profile]
        self.stop_words = _stop_words(self.profile)
        if self.profile.custom_recognizers:
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
        offsets, texts = zip(*_chunks(text, max_words=self.max_words)) if text.strip() else ((), ())
        for off, ents in zip(offsets, self.gliner.batch_predict_entities(
                list(texts), self.labels, threshold=self.threshold, batch_size=self.batch_size)):
            spans += [Span(off + e["start"], off + e["end"], e["text"],
                           self.label2type[e["label"]], e["score"], "gliner") for e in ents]
        for r in self.presidio.analyze(text=text, language="en"):
            if r.entity_type in PRESIDIO_MAP:
                t = PRESIDIO_MAP[r.entity_type]
                if self.fine_dem and t == "DEM":
                    continue   # fine-dem: GLiNER's learned fine leaves own demographics; drop Presidio's
                               # coarse NRP->DEM (keeps relabel_dem training/eval-only, inference pure-model).
                rec = (r.recognition_metadata or {}).get("recognizer_name", "")
                src = "presidio" if rec == "SpacyRecognizer" else "presidio-pattern"
                spans.append(Span(r.start, r.end, text[r.start:r.end], t, r.score, src))
        spans = [s for s in spans  # pure symbol/emoji spans or bare stop words: never identifiers
                 if re.search(r"[A-Za-z0-9]", s.text) and s.text.lower() not in self.stop_words]
        spans = _dedupe(spans)
        if self.profile.negative_filter:
            spans = _apply_negative_filter(spans)
        return spans


def _dedupe(spans: list[Span]) -> list[Span]:
    """Overlap resolution. Same-type overlaps: keep the widest (extent disagreement over one
    entity). Cross-type conflicts: higher score wins; pattern-based Presidio hits (fixed regex
    scores 0.4-0.6, not comparable to GLiNER probabilities) get an effective floor of 0.9.
    """
    def eff(s: Span) -> float:
        return max(s.score, 0.9) if s.source == "presidio-pattern" else s.score

    within_type: list[Span] = []
    for s in sorted(spans, key=lambda s: (-(s.end - s.start), -s.score, s.start)):
        if not any(s.type == o.type and s.start < o.end and o.start < s.end for o in within_type):
            within_type.append(s)
    out: list[Span] = []
    for s in sorted(within_type, key=lambda s: (-eff(s), -(s.end - s.start), s.start)):
        if not any(s.start < o.end and o.start < s.end for o in out):
            out.append(s)
    return sorted(out, key=lambda s: s.start)


def _apply_negative_filter(spans: list[Span]) -> list[Span]:
    """Semantic-gate the noise-filter types: drop margin/deny-list negatives, apply layer-2
    retypes, keep the rest. Types outside _NOISE_FILTER_TYPES bypass the gate. Fail-opens to
    keep when gate artifacts are absent (== the old is_noise_span filter via the deny-list
    layer). span_gate is imported lazily so a non-clinical profile never pulls numpy at load."""
    gated = [s for s in spans if s.type in _NOISE_FILTER_TYPES]
    if not gated:
        return spans
    from cloak import span_gate
    from cloak.profile_match import span_key
    decisions = span_gate.gate_spans([(s.text, s.type) for s in gated], "production")
    out: list[Span] = []
    for s in spans:
        if s.type not in _NOISE_FILTER_TYPES:
            out.append(s)
            continue
        d = decisions.get(span_key(s.text, s.type))
        if d is None or d.action == "keep":
            out.append(s)
        elif d.action == "retype" and d.new_type:
            out.append(replace(s, type=d.new_type))
        # d.action == "drop": omit
    return out


def coref_chains(text: str, spans: list[Span]) -> list[Span]:
    """Attach chain ids by surface aliasing. Same-type spans chain only when one surface
    contains the other as whole tokens ("Anna Smith" ~ "Anna") or their first tokens match
    ("Anna Smith" ~ "Anna S."). A bare single-token mention that is a non-first token of an
    existing member (bare "Smith") joins the MOST RECENT matching chain. Any-token overlap
    is deliberately not enough: it merged distinct people sharing a surname into one
    placeholder.

    ponytail: string-alias coref — fastcoref 2.1.6 is incompatible with transformers 5.12
    (FCorefModel hits removed modeling internals). Upgrade to a real coref model for the TAB
    pass, where nominal anaphora ("the applicant") matters.
    """
    def toks(surface: str) -> list[str]:
        return [t for t in surface.lower().split() if len(t) > 2]

    def aliases(a: list[str], b: list[str]) -> bool:
        if not a or not b:
            return False
        # containment requires BOTH sides multi-token: a bare stored member ("smith") must not
        # become a containment magnet that transitively re-merges distinct people. Single-token
        # aliasing is handled exclusively by first-token match and the bare-mention clause below.
        if min(len(a), len(b)) >= 2:
            sa, sb = set(a), set(b)
            if sa <= sb or sb <= sa:
                return True
        return a[0] == b[0]

    chains: list[tuple[str, list[list[str]]]] = []  # (type, member token lists)
    for s in sorted(spans, key=lambda s: s.start):
        st = toks(s.text)
        s.chain = -1
        for ci in reversed(range(len(chains))):      # most recent chain wins
            ctype, members = chains[ci]
            if ctype != s.type:
                continue
            if any(aliases(st, m) for m in members) or (
                    len(st) == 1 and any(st[0] in m for m in members)):
                s.chain = ci
                members.append(st)
                break
        if s.chain < 0:
            chains.append((s.type, [st]))
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
