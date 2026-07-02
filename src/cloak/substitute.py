"""Substitutor: doc_orig -> (doc_p, R).

Direct identifiers -> typed placeholders numbered per alias chain.
Quasi-identifiers -> generalization lattice walk, most-specific-first, accepting the
first level whose MTI guess-back risk < tau (tau is the privacy knob).
R (substitution record) stays client-side and drives extraction.
"""
import re

from cloak.detect import Detector, Span, coref_chains
from cloak.lattice import TYPE_LABEL, lattice_for
from cloak.probe import guess_back_risk

DIRECT_TYPES = {"PERSON", "CODE"}


def _is_role_phrase(text: str) -> bool:
    """True if a lowercase PERSON hit is a common-noun role ("patient", "the applicant"),
    not a name. Privacy-safe bias: keep PERSON unless clearly a role, so lowercase names in
    dialogue/informal text ("martha") still get placeholders instead of generalizing.

    ponytail: WordNet single-token noun + article prefix. A name that is also a common noun
    ("Bill", "Rose", "May") lowercased still misroutes to DEM — inherent lowercase ambiguity;
    upgrade with a names gazetteer if it bites.
    """
    from nltk.corpus import wordnet as wn
    t = text.strip().lower()
    if re.match(r"(?:the|a|an|this|that|my|his|her|their)\b", t):  # "the applicant", "a nurse"
        return True
    toks = re.findall(r"[a-z]+", t)
    if len(toks) != 1:  # multi-word bare phrase is a proper name ("mary jane") -> keep PERSON
        return False
    return bool(wn.synsets(toks[0], pos=wn.NOUN))


def _sentence_around(text: str, start: int, end: int) -> str:
    lo = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    hi_candidates = [i for i in (text.find(".", end), text.find("\n", end)) if i != -1]
    hi = min(hi_candidates) + 1 if hi_candidates else len(text)
    return text[lo:hi].strip()


def substitute(text: str, spans: list[Span], tau: float = 0.02) -> tuple[str, list[dict]]:
    """Returns (doc_p, R). Spans must be non-overlapping (Detector dedupes)."""
    # generic temporals ("daily", "these days", "summer") are not identifiers: substituting
    # them wrecks readability for zero privacy; only dated/aged DATETIMEs are processed
    spans = [s for s in spans if not (
        s.type == "DATETIME" and not re.search(
            r"\d|january|february|march|april|may|june|july|august|september|october"
            r"|november|december|year[s]?[\s-]old|\b(?:last|next|previous|past)\b",
            s.text, re.IGNORECASE))]
    for s in spans:  # a lowercase "PERSON" role noun (lawyer, patient) generalizes; a
        if s.type == "PERSON" and s.text[0].islower() and _is_role_phrase(s.text):  # name stays
            s.type = "DEM"
    spans = coref_chains(text, spans)
    counters: dict[str, int] = {}
    chain_ph: dict[int, str] = {}
    record = []
    out = text
    for s in sorted(spans, key=lambda s: -s.start):  # right-to-left keeps offsets valid
        entry = {"start": s.start, "end": s.end, "surface": s.text, "type": s.type,
                 "chain": s.chain, "score": s.score}
        if s.type in DIRECT_TYPES:
            ph = chain_ph.get(s.chain)
            if ph is None:
                counters[s.type] = counters.get(s.type, 0) + 1
                ph = f"<{s.type}_{counters[s.type]}>"
                chain_ph[s.chain] = ph
            entry.update(action="placeholder", replacement=ph, risk=0.0)
        else:
            sent = _sentence_around(text, s.start, s.end)
            lattice = lattice_for(s.text, s.type, sent)
            # candidate must not carry the original's numbers or proper names
            distinctive = set(re.findall(r"\d[\d,.]*\d|\d", s.text)) | \
                {w.lower() for w in re.findall(r"\b[A-Z][a-z]{2,}\b", s.text)}
            lattice = [c for c in lattice
                       if not distinctive & (set(re.findall(r"\d[\d,.]*\d|\d", c)) |
                                             set(re.findall(r"\w{3,}", c.lower())))] \
                or [TYPE_LABEL.get(s.type, "something")]
            chosen, risk = lattice[-1], None
            for cand in lattice:
                cand_sent = sent.replace(s.text, cand) if s.text in sent else cand
                risk = guess_back_risk(cand_sent, s.text, cand)
                if risk < tau:
                    chosen = cand
                    break
            prev = text[:s.start].rstrip()
            sent_start = not prev or prev[-1] in ".!?\n"
            chosen = (chosen[0].upper() if sent_start else chosen[0].lower()) + chosen[1:]
            entry.update(action="generalize", replacement=chosen, lattice=lattice,
                         risk=round(risk, 4) if risk is not None else None)
        out = out[:s.start] + entry["replacement"] + out[s.end:]
        record.append(entry)
    out = re.sub(r"\b([Aa]n?|[Tt]he) (?=(?:an?|the)\b)", "", out)  # "a a person", "the a structure"
    out = re.sub(r"\b[Ii]n (?=in\b)", "", out)                    # "in in the spring"
    return out, record[::-1]


class Substitutor:
    """Convenience wrapper: detector + substitute at a fixed tau."""

    def __init__(self, tau: float = 0.02, **det_kw):
        self.tau = tau
        self.det = Detector(**det_kw)

    def __call__(self, text: str) -> tuple[str, list[dict]]:
        return substitute(text, self.det.detect(text), tau=self.tau)


if __name__ == "__main__":
    sub = Substitutor()
    text = ("Sarah Johnson is a cardiologist at the university hospital in Oslo. "
            "Sarah moved from Bergen in 2019 and earns 120,000 dollars. "
            "Her case ref is 36110/97.")
    doc_p, R = sub(text)
    print(doc_p)
    for r in R:
        print(f"  {r['action']:11s} {r['surface']!r:22s} -> {r['replacement']!r} "
              f"(risk={r.get('risk')})")
    assert "Sarah" not in doc_p and "36110/97" not in doc_p
    assert "Oslo" not in doc_p and "Bergen" not in doc_p
    assert "120,000" not in doc_p, doc_p  # income is a SynthPAI gold attribute
    ph = [r["replacement"] for r in R if r["surface"].startswith("Sarah")]
    assert len(set(ph)) == 1, ph  # same chain -> same placeholder

    # lowercase-name routing: names stay PERSON, role nouns generalize
    assert not _is_role_phrase("martha") and not _is_role_phrase("dmitri")
    assert not _is_role_phrase("mary jane")  # multi-word bare name
    assert _is_role_phrase("patient") and _is_role_phrase("nurse")
    assert _is_role_phrase("the applicant") and _is_role_phrase("a cardiologist")
    print("substitute.py self-check OK")
