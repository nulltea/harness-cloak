"""Substitutor: doc_orig -> (doc_p, R).

Direct identifiers -> typed placeholders numbered per alias chain.
Quasi-identifiers -> generalization lattice walk, most-specific-first, accepting the
first level whose MTI guess-back risk < tau (tau is the privacy knob).
R (substitution record) stays client-side and drives extraction.
"""
import re
from dataclasses import replace

from cloak.detect import Detector, Span, coref_chains, relabel_dem
from cloak.lattice import TYPE_LABEL, lattice_for, NO_PREPASS
from cloak.lattice_profiles import lookup_entry
from cloak.probe import walk_risk
from cloak.profile_match import PROFILE_BACKED_TYPES, match_spans_batch, span_key
from cloak.runtime_types import DIRECT_TYPES, PLACEHOLDER_RE, placeholder_token, placeholder_type_token

DIRECT_TYPES = set(DIRECT_TYPES)
_EXPLICIT_NAME_LABELS = frozenset({"name", "first name", "last name"})


def _is_role_phrase(text: str) -> bool:
    """True if a lowercase PERSON hit is a common-noun role ("patient", "the applicant"),
    not a name. Privacy-safe bias: keep PERSON unless clearly a role, so lowercase names in
    dialogue/informal text ("martha") still get placeholders instead of generalizing.

    ponytail: WordNet single-token noun + article prefix. A name that is also a common noun
    ("Bill", "Rose", "May") lowercased can still misroute to a fine demographic leaf —
    inherent lowercase ambiguity; upgrade with a names gazetteer if it bites.
    """
    from nltk.corpus import wordnet as wn
    t = text.strip().lower()
    if re.match(r"(?:the|a|an|this|that|my|his|her|their)\b", t):  # "the applicant", "a nurse"
        return True
    toks = re.findall(r"[a-z]+", t)
    if len(toks) != 1:  # multi-word bare phrase is a proper name ("mary jane") -> keep PERSON
        return False
    return bool(wn.synsets(toks[0], pos=wn.NOUN))


def prepare_spans_for_substitution(
    text: str,
    spans: list[Span],
    *,
    reject_demographic_other: bool = False,
    min_health_condition_score: float | None = None,
) -> tuple[list[Span], list[dict]]:
    prepared: list[Span] = []
    rejected: list[dict] = []
    for source_span in spans:
        span = replace(source_span)
        # Low-confidence health-conditions are exam-findings/modifiers, not
        # diagnoses (edema/erythema/immunocompromised/acute exacerbation score
        # well below real diagnoses on GLiNER's `condition` label). Reject them
        # before they become controlled decisions rather than freeze noise the
        # relation teacher then anchors on. Only health-condition is gated;
        # exp:detector-finding-vs-diagnosis-separation.
        if (
            min_health_condition_score is not None
            and span.type == "health-condition"
            and (span.score or 0.0) < min_health_condition_score
        ):
            rejected.append({
                "start": span.start,
                "end": span.end,
                "surface": span.text,
                "source": span.source,
                "raw_label": span.raw_label,
                "recognizer": span.recognizer,
                "score": span.score,
                "status": "post_detection_rejected",
                "reason": "qa_v2_low_confidence_health_condition",
                "min_health_condition_score": min_health_condition_score,
            })
            continue
        if (
            span.type == "PERSON"
            and span.text[0].islower()
            and span.raw_label not in _EXPLICIT_NAME_LABELS
            and _is_role_phrase(span.text)
        ):
            proposed = relabel_dem(span.text)
            if reject_demographic_other and proposed == "demographic-other":
                rejected.append({
                    "start": span.start,
                    "end": span.end,
                    "surface": span.text,
                    "source": span.source,
                    "raw_label": span.raw_label,
                    "recognizer": span.recognizer,
                    "score": span.score,
                    "status": "post_detection_rejected",
                    "reason": "qa_v2_forbidden_demographic_other",
                    "proposed_runtime_type": proposed,
                })
                continue
            span.type = proposed
        prepared.append(span)
    return prepared, rejected


def _sentence_around(text: str, start: int, end: int) -> str:
    lo = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    hi_candidates = [i for i in (text.find(".", end), text.find("\n", end)) if i != -1]
    hi = min(hi_candidates) + 1 if hi_candidates else len(text)
    return text[lo:hi].strip()


def _starts_with_vowel_sound(word: str) -> bool:
    word = word.lower()
    if word.startswith(("honest", "honor", "hour", "heir")):
        return True
    if word.startswith(("uni", "use", "user", "euro", "one")):
        return False
    return bool(word) and word[0] in "aeiou"


def _fix_indefinite_articles(text: str) -> str:
    def repl(match: re.Match) -> str:
        article, word = match.groups()
        fixed = "an" if _starts_with_vowel_sound(word) else "a"
        if article[0].isupper():
            fixed = fixed.title()
        return f"{fixed} {word}"

    return re.sub(r"\b([Aa]n?)\s+([A-Za-z][A-Za-z-]*)", repl, text)


def freeze_policy_free_candidates(text: str, spans: list[Span]) -> list[dict]:
    """Freeze detector/profile/lattice facts without executing a privacy policy.

    This is the QA-v2/Ranker-v2 source path. It deliberately never calls
    ``walk_risk`` and emits no chosen replacement, risk, tau outcome, or exhausted
    state. The legacy ``substitute`` path below remains unchanged.
    """
    spans = [s for s in spans if not (
        s.type == "DATETIME" and not re.search(
            r"\d|january|february|march|april|may|june|july|august|september|october"
            r"|november|december|year[s]?[\s-]old|\b(?:last|next|previous|past)\b",
            s.text, re.IGNORECASE,
        )
    )]
    spans = coref_chains(text, spans)
    items = [
        (s.text, s.type, _sentence_around(text, s.start, s.end))
        for s in spans if s.type in PROFILE_BACKED_TYPES
    ]
    proposals = match_spans_batch(items) if items else {}
    by_surface: dict[tuple[str, str], dict] = {}
    records = []
    for span in sorted(spans, key=lambda row: (row.start, row.end)):
        detector_provenance = (
            dict(span.detector_provenance)
            if span.detector_provenance is not None
            else {
                "source": span.source, "raw_label": span.raw_label,
                "recognizer": span.recognizer, "score": span.score,
            }
        )
        row = {
            "start": span.start, "end": span.end, "surface": span.text,
            "type": span.type, "chain": span.chain, "score": span.score,
            "detector_provenance": detector_provenance,
        }
        surface_key = (span.type, span.text.casefold())
        previous = by_surface.get(surface_key)
        if previous is not None:
            row.update(previous)
            records.append(row)
            continue
        if span.type in DIRECT_TYPES:
            # Forced protection, not a ranker action: every V2 render uses the
            # typed placeholder and no KEEP action is synthesized downstream.
            frozen = {"lattice": [], "forced_placeholder": True, "uncontrolled": False}
        else:
            sentence = _sentence_around(text, span.start, span.end)
            key = span_key(span.text, span.type)
            proposal = proposals[key] if key in proposals else NO_PREPASS
            lattice = lattice_for(span.text, span.type, sentence, proposal=proposal)
            match = proposals.get(key)
            distinctive = set(re.findall(r"\d[\d,.]*\d|\d", span.text)) | {
                word.lower() for word in re.findall(r"\b[A-Z][a-z]{2,}\b", span.text)
            }
            lattice = [
                candidate for candidate in lattice
                if PLACEHOLDER_RE.fullmatch(candidate)
                or not distinctive & (
                    set(re.findall(r"\d[\d,.]*\d|\d", candidate))
                    | set(re.findall(r"\w{3,}", candidate.lower()))
                )
            ] or [
                placeholder_token(span.type, 1)
                if span.type not in TYPE_LABEL else TYPE_LABEL.get(span.type, "something")
            ]
            real_levels = [
                candidate for candidate in lattice if not PLACEHOLDER_RE.fullmatch(candidate)
            ]
            profile_entry = lookup_entry(span.text, span.type)
            frozen = {
                "lattice": lattice,
                "uncontrolled": bool(
                    span.type in PROFILE_BACKED_TYPES and not real_levels and profile_entry is None
                ),
            }
            if match is not None:
                frozen["match"] = (
                    {"kind": "exact", "entry": match.entry}
                    if match.kind == "exact" else
                    {"kind": "semantic", "entry": match.entry,
                     "similarity": round(match.similarity, 3),
                     "nli": round(match.nli, 3) if match.nli is not None else None}
                )
                frozen["profile_match"] = {
                    "outcome": match.kind, "reason": f"{match.kind}_entry",
                    "entry": match.entry,
                }
            elif span.type in PROFILE_BACKED_TYPES:
                frozen["profile_match"] = {
                    "outcome": "abstained", "reason": "no_certified_match",
                }
        by_surface[surface_key] = frozen
        row.update(frozen)
        records.append(row)
    return records


def substitute(text: str, spans: list[Span], tau: float = 0.02) -> tuple[str, list[dict]]:
    """Returns (doc_p, R). Spans must be non-overlapping (Detector dedupes)."""
    spans, _rejected = prepare_spans_for_substitution(text, spans)
    # generic temporals ("daily", "these days", "summer") are not identifiers: substituting
    # them wrecks readability for zero privacy; only dated/aged DATETIMEs are processed
    spans = [s for s in spans if not (
        s.type == "DATETIME" and not re.search(
            r"\d|january|february|march|april|may|june|july|august|september|october"
            r"|november|december|year[s]?[\s-]old|\b(?:last|next|previous|past)\b",
            s.text, re.IGNORECASE))]
    spans = coref_chains(text, spans)
    # batched matcher pre-pass: one embed batch + wave-batched NLI for the whole doc
    # (docs/specs/substitutor-profile-match-retrieve-verify.md, Efficiency)
    items = [(s.text, s.type, _sentence_around(text, s.start, s.end))
             for s in spans if s.type in PROFILE_BACKED_TYPES]
    proposals = match_spans_batch(items) if items else {}
    counters: dict[str, int] = {}
    chain_ph: dict[int, str] = {}
    used: dict[str, str] = {}          # replacement canon -> surface canon (injectivity of R)
    by_surface: dict[str, dict] = {}   # surface canon -> {replacement, risk, ...} (repeat reuse)
    record = []
    out = text

    def _typed_placeholder(s) -> str:
        tok = placeholder_type_token(s.type)
        counters[tok] = counters.get(tok, 0) + 1
        return placeholder_token(s.type, counters[tok])

    # Reserve the exact surfaces that will be KEPT (unprofiled generalizable detections) so no
    # generalization fill below can equal one. A shared string would make replacement-keyed
    # un-perturb restore the kept span to the generalized entity's source, corrupting out_final
    # (e.g. "acid reflux"->"gastrointestinal condition" colliding with a kept literal
    # "gastrointestinal condition"). A generalization whose fill collides falls to placeholder.
    for s in spans:
        if s.type in DIRECT_TYPES or s.type not in PROFILE_BACKED_TYPES:
            continue
        _sent = _sentence_around(text, s.start, s.end)
        _lat = lattice_for(s.text, s.type, _sent,
                           proposal=proposals.get(span_key(s.text, s.type), NO_PREPASS))
        if (not any(not PLACEHOLDER_RE.fullmatch(c) for c in _lat)
                and lookup_entry(s.text, s.type) is None):
            used.setdefault(s.text.lower(), f"__keep__:{s.text.lower()}")

    for s in sorted(spans, key=lambda s: -s.start):  # right-to-left keeps offsets valid
        detector_provenance = (
            dict(s.detector_provenance)
            if s.detector_provenance is not None
            else {
                "source": s.source,
                "raw_label": s.raw_label,
                "recognizer": s.recognizer,
                "score": s.score,
            }
        )
        entry = {"start": s.start, "end": s.end, "surface": s.text, "type": s.type,
                 "chain": s.chain, "score": s.score,
                 "detector_provenance": detector_provenance}
        skey = s.text.lower()
        if s.type in DIRECT_TYPES:
            ph = chain_ph.get(s.chain)
            if ph is None:
                ph = _typed_placeholder(s)
                chain_ph[s.chain] = ph
            entry.update(action="placeholder", replacement=ph, risk=0.0)
        elif skey in by_surface:  # repeat mention: reuse its own replacement (still injective)
            entry.update(by_surface[skey])
        else:
            sent = _sentence_around(text, s.start, s.end)
            k = span_key(s.text, s.type)
            prop = proposals[k] if k in proposals else NO_PREPASS
            lattice = lattice_for(s.text, s.type, sent, proposal=prop)
            # Is this a controllable entity at all? Profiled = it has a lattice-profile entry
            # OR the prepass produced a real (non-placeholder) generalization for it. An
            # UNPROFILED detection (neither) is detector-only evidence -- keep it EXACT rather
            # than mint a junk placeholder for a possible false positive. A profiled entity that
            # merely FAILED to resolve this mention (abstain / all-over-tau) is NOT unprofiled:
            # it must still be hidden. (user rule 2026-07-16; matches freeze controlled=False.)
            profiled = (
                any(not PLACEHOLDER_RE.fullmatch(c) for c in lattice)
                or lookup_entry(s.text, s.type) is not None
            )
            m = proposals.get(k)
            if m is not None:
                entry["match"] = ({"kind": "exact", "entry": m.entry} if m.kind == "exact" else
                                  {"kind": "semantic", "entry": m.entry,
                                   "similarity": round(m.similarity, 3),
                                   "nli": round(m.nli, 3) if m.nli is not None else None})
            # candidate must not carry the original's numbers or proper names
            distinctive = set(re.findall(r"\d[\d,.]*\d|\d", s.text)) | \
                {w.lower() for w in re.findall(r"\b[A-Z][a-z]{2,}\b", s.text)}
            lattice = [c for c in lattice
                       if PLACEHOLDER_RE.fullmatch(c) or
                       not distinctive & (set(re.findall(r"\d[\d,.]*\d|\d", c)) |
                                           set(re.findall(r"\w{3,}", c.lower())))] \
                or [placeholder_token(s.type, 1) if s.type not in TYPE_LABEL
                    else TYPE_LABEL.get(s.type, "something")]
            chosen, risk = None, None
            for cand in lattice:
                if PLACEHOLDER_RE.fullmatch(cand):
                    continue
                if used.get(cand.lower(), skey) != skey:
                    continue  # claimed by a different surface (injectivity)
                cand_sent = sent.replace(s.text, cand) if s.text in sent else cand
                r = walk_risk(cand_sent, s.text, cand, s.type)
                if r < tau:
                    chosen, risk = cand, r
                    break
            if chosen is None and not profiled and s.type in PROFILE_BACKED_TYPES:
                # UNPROFILED detection of a GENERALIZABLE type (one that lattice_profiles backs:
                # drug/health-condition/medical-procedure/org/loc/...): no real level exists, so
                # leave it EXACT and uncontrolled -- the ranker never rewrites it, it gets no
                # action/risk/privacy credit, and its text survives every scored render (a literal
                # answer can rest on it). Prevents detector false positives from becoming
                # destructive junk placeholders that contaminate utility scoring and training.
                # NON-generalizable types (PERSON/CODE via DIRECT_TYPES already, plus gender/age/
                # marital-status/... not in PROFILE_BACKED_TYPES) must still be placeholdered.
                entry.update(action="keep", replacement=s.text, risk=0.0,
                             lattice=lattice, uncontrolled=True)
            elif chosen is None:
                # exhausted (every level over tau or claimed): generic typed placeholder —
                # risk 0 by construction; tau becomes a hard guarantee, never the old
                # over-budget floor. Spec §3.3-2.
                entry.update(action="placeholder", replacement=_typed_placeholder(s),
                             risk=0.0, lattice=lattice, exhausted=True)
            else:
                prev = text[:s.start].rstrip()
                sent_start = not prev or prev[-1] in ".!?\n"
                chosen = (chosen[0].upper() if sent_start else chosen[0].lower()) + chosen[1:]
                used[chosen.lower()] = skey
                entry.update(action="generalize", replacement=chosen, lattice=lattice,
                             risk=round(risk, 4))
            by_surface[skey] = {k2: entry[k2] for k2 in
                                ("action", "replacement", "risk", "lattice", "match",
                                 "uncontrolled")
                                if k2 in entry}
        out = out[:s.start] + entry["replacement"] + out[s.end:]
        record.append(entry)
    out = re.sub(r"\b([Aa]n?|[Tt]he) (?=(?:an?|the)\b)", "", out)  # "a a person", "the a structure"
    out = re.sub(r"\b[Ii]n (?=in\b)", "", out)                    # "in in the spring"
    out = _fix_indefinite_articles(out)
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
              f"(risk={r.get('risk')}{', EXHAUSTED' if r.get('exhausted') else ''})")
    assert "Sarah" not in doc_p and "36110/97" not in doc_p
    assert "Oslo" not in doc_p and "Bergen" not in doc_p
    assert "120,000" not in doc_p, doc_p  # income is a SynthPAI gold attribute
    ph = [r["replacement"] for r in R if r["surface"].startswith("Sarah")]
    assert len(set(ph)) == 1, ph  # same chain -> same placeholder

    # injectivity of R: a replacement maps back to one entity — one surface for
    # generalizations, one coref chain for placeholders (same chain shares its token)
    rep_owner = {}
    for r in R:
        owner = (r["surface"].lower() if r["action"] in ("generalize", "keep")
                 else f"chain:{r['chain']}")
        prev = rep_owner.setdefault(r["replacement"].lower(), owner)
        assert prev == owner, (r["replacement"], prev, owner)
    # tau is a hard guarantee: every shipped generalization is under budget
    assert all(r["risk"] < sub.tau for r in R if r["action"] == "generalize"), R
    # exhausted spans ship as typed placeholders, never over-tau floors
    assert all(PLACEHOLDER_RE.fullmatch(r["replacement"])
               for r in R if r.get("exhausted")), R

    # lowercase-name routing: names stay PERSON, role nouns generalize
    assert not _is_role_phrase("martha") and not _is_role_phrase("dmitri")
    assert not _is_role_phrase("mary jane")  # multi-word bare name
    assert _is_role_phrase("patient") and _is_role_phrase("nurse")
    assert _is_role_phrase("the applicant") and _is_role_phrase("a cardiologist")
    print("substitute.py self-check OK")
