"""Anonymity-set counts for lattice levels — structural risk, zero models at inference.

count = number of candidate values AT THE ORIGINAL'S GRANULARITY consistent with the fill
(k-anonymity's consistency set transplanted to span substitution; design:
docs/research/inference-risk-enforcement.md). Per-type floors replace the walk_risk
tau-mask: legal iff aset >= k_floors[type] (placeholder always legal, keep-original has
aset 1 -> legal only for user-waived types). Units need only be consistent WITHIN a type —
floors are per-type, so city-counts and day-windows never compare.

Fail-closed: a specific-looking but unparseable fill counts 1 (illegal under any floor > 1,
same conservative direction as the probe's unpooled-type rule). The six TYPE_LABEL coarse
fills are known-generic -> GENERIC, checked first.

Two modes. strict=True (CERTIFYING — legality, artifact annotation, decode-time checks):
a parse miss fails closed, never falls through to a broader lookup; in particular the
WordNet last-word fallback is off ("a coastal Norwegian city" -> no full-phrase synset ->
1.0, NOT the count of all cities). Deterministic is not un-gameable: permissive fallbacks
fail OPEN on adversarial phrasings, and a generation loop would mine them. strict=False is
for diagnostics only.
"""
import re
from collections import Counter
from decimal import Decimal
from functools import lru_cache

from cloak.lattice import CONTINENTS, TYPE_LABEL, _MONTHS, _load_geo

GENERIC = 1e9

# provisional floors — Task 2 replaces these with values calibrated on the measured
# count-vs-attacker table (empirical-honesty rule: never ship invented numbers)
K_FLOORS = {"LOC": 50.0, "ORG": 50.0, "DATETIME": 30.0,
            "DEM": 1.0, "QUANTITY": 1.0, "MISC": 1.0, "OTHER": 1.0}

_geo_counts_cache = None


def _geo_counts():
    """Reverse count indexes over cities500: unique city names per admin1 region name,
    per country name, per continent name; plus countries per continent."""
    global _geo_counts_cache
    if _geo_counts_cache is None:
        cities, admin1, countries = _load_geo()
        by_region, by_country, by_cont, cont_countries = (Counter(), Counter(),
                                                          Counter(), Counter())
        for ccode, a1, _pop in cities.values():
            cn = countries.get(ccode)
            if not cn:
                continue
            by_country[cn["name"].lower()] += 1
            cont = CONTINENTS.get(cn["continent"])
            if cont:
                by_cont[cont.lower()] += 1
            region = admin1.get(f"{ccode}.{a1}")
            if region:
                by_region[region.lower()] += 1
        for cc in countries.values():
            cont = CONTINENTS.get(cc["continent"])
            if cont:
                cont_countries[cont.lower()] += 1
        _geo_counts_cache = (by_region, by_country, by_cont, cont_countries)
    return _geo_counts_cache


def _loc_count(fill: str) -> float | None:
    by_region, by_country, by_cont, cont_countries = _geo_counts()
    m = re.fullmatch(r"a city in (.+)", fill.strip(), re.IGNORECASE)
    if m:
        key = m.group(1).lower()
        n = by_cont.get(key) or by_country.get(key) or by_region.get(key)
        return float(n) if n else None
    m = re.fullmatch(r"a country in (.+)", fill.strip(), re.IGNORECASE)
    if m:
        n = cont_countries.get(m.group(1).lower())
        return float(n) if n else None
    return None


_MONTH_RE = "|".join(_MONTHS)
_AGE_FILL = {"teenaged": 7.0}  # plus "X-something" -> 10 (a decade of ages)


def _date_granularity_days(original: str) -> float | None:
    t = original.strip().lower()
    has_year = re.search(r"\b(19|20)\d{2}\b", t)
    has_month = re.search(rf"\b({_MONTH_RE})\b", t)
    has_day = re.search(r"\b([0-3]?\d)(st|nd|rd|th)?\b", t)
    if has_month and has_year and has_day and not re.fullmatch(r"(19|20)\d{2}", t):
        return 1.0       # "12 March 2019"
    if has_month and has_year:
        return 30.0      # "March 2019"
    if has_month:
        return 30.0      # "March", "November 7" (day unknown-year: month-grain)
    if has_year:
        return 365.0     # "2019"
    return None          # relative phrases ("last year"): no absolute granularity


def _date_window_days(fill: str) -> float | None:
    f = fill.strip().lower()
    if re.fullmatch(r"the (early|mid|late) (19|20)\d0s", f):
        return 3652 / 3
    if re.fullmatch(r"the (19|20)\d0s", f):
        return 3652.0
    if re.fullmatch(rf"({_MONTH_RE}) (19|20)\d{{2}}", f):
        return 30.0
    if re.fullmatch(r"the (winter|spring|summer|autumn|fall)", f):
        return 91.0
    return None


def _datetime_count(fill: str, original: str) -> float | None:
    f = fill.strip().lower()
    if f in _AGE_FILL:
        return _AGE_FILL[f]
    if re.fullmatch(r"\w+-something", f):
        return 10.0
    if f in ("at some point", "some time ago"):
        return GENERIC
    window = _date_window_days(fill)
    if window is None:
        return None
    gran = _date_granularity_days(original)
    if gran is None:
        return None
    return max(window / gran, 1.0)


def _step(v: float) -> float:
    """Magnitude of the least-significant digit: 40 -> 10, 45 -> 1, 2.5 -> 0.1."""
    d = Decimal(str(v)).normalize()
    return float(Decimal(1).scaleb(d.as_tuple().exponent))


def _quantity_count(fill: str, original: str) -> float | None:
    m = re.search(r"between ([\d,\.]+) and ([\d,\.]+)", fill, re.IGNORECASE)
    om = re.search(r"[\d][\d,.]*", original)
    if not (m and om):
        return None
    try:
        lo, hi = (float(m.group(i).replace(",", "")) for i in (1, 2))
        v = float(om.group().replace(",", ""))
    except ValueError:
        return None
    return max((hi - lo) / _step(v) + 1, 1.0)


@lru_cache(maxsize=4096)
def _wn_leaf_count(phrase: str, strict: bool = False) -> float | None:
    """Leaves of the hyponym closure under the phrase's first noun synset (the same
    sense-selection policy as lattice.wordnet_chain). strict: full-phrase synset only —
    the last-word fallback over-counts ("...city" -> all cities), a fail-open a
    generation loop would mine."""
    from nltk.corpus import wordnet as wn
    p = re.sub(r"^(an?|the) ", "", phrase.lower().strip())
    syns = wn.synsets(p.replace(" ", "_"), pos=wn.NOUN)
    if not syns and not strict and " " in p:
        syns = wn.synsets(p.split()[-1], pos=wn.NOUN)  # diagnostic-only fallback
    if not syns:
        return None
    seen, stack, leaves = set(), [syns[0]], 0
    while stack:
        s = stack.pop()
        if s.name() in seen:
            continue
        seen.add(s.name())
        subs = s.hyponyms() + s.instance_hyponyms()
        if subs:
            stack.extend(subs)
        else:
            leaves += 1
    return float(max(leaves, 1))


@lru_cache(maxsize=8192)
def aset_count(fill: str, span_type: str, original: str, strict: bool = False) -> float:
    """Anonymity-set size of `fill` for a span of `span_type` whose original text is
    `original`. 1.0 for keep-original and for fail-closed unparseable fills; GENERIC for
    type-label coarse fills. strict=True for anything that certifies legality."""
    if fill.lower().strip() == original.lower().strip():
        return 1.0
    if fill.lower().strip() in {v.lower() for v in TYPE_LABEL.values()}:
        return GENERIC
    got = None
    if span_type == "LOC":
        got = _loc_count(fill) or _wn_leaf_count(fill, strict)
    elif span_type == "DATETIME":
        got = _datetime_count(fill, original)
    elif span_type == "QUANTITY":
        got = _quantity_count(fill, original)
    else:  # DEM / ORG / MISC / OTHER — WordNet lattices
        got = _wn_leaf_count(fill, strict)
    return got if got else 1.0  # ponytail: fail-closed; per-surface overrides if too strict


if __name__ == "__main__":
    assert aset_count("Oslo", "LOC", "Oslo") == 1.0                      # keep-original
    n_no = aset_count("a city in Norway", "LOC", "Oslo")
    n_eu = aset_count("a city in Europe", "LOC", "Oslo")
    assert 4 < n_no < n_eu, (n_no, n_eu)                                 # ordering
    assert aset_count("a place", "LOC", "Oslo") == GENERIC               # TYPE_LABEL
    d_mo = aset_count("March 2019", "DATETIME", "12 March 2019")
    d_dec = aset_count("the 2010s", "DATETIME", "12 March 2019")
    assert 1 < d_mo < d_dec, (d_mo, d_dec)
    assert aset_count("thirty-something", "DATETIME", "34") == 10.0
    q = aset_count("between 20 and 80 milligrams", "QUANTITY", "40 milligrams")
    assert q == 7.0, q                                                    # step 10
    w_hd = aset_count("a heart disease", "DEM", "hypertension")
    w_d = aset_count("a disease", "DEM", "hypertension")
    assert 1 < w_hd < w_d, (w_hd, w_d)
    assert aset_count("xyzzy blorp", "DEM", "hypertension") == 1.0        # fail-closed
    # strict mode: no last-word fallback — a narrowing phrase must fail CLOSED, while
    # permissive mode may fail OPEN (that asymmetry is why legality always uses strict)
    assert aset_count("a coastal Norwegian city", "LOC", "Bergen", strict=True) == 1.0
    assert aset_count("a coastal Norwegian city", "LOC", "Bergen") > 1.0  # diagnostic
    print("anonymity.py self-check OK")
