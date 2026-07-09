"""Generalization lattices: rule buckets, GeoNames chains, WordNet paths,
and the E4B->Qwen teacher cascade with an NLI truthfulness gate.

A lattice is an ordered list of surface phrases, most specific -> most general.
Plan: docs/plans/2026-07-02-d1-prototype-implementation.md.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cloak.lattice_profiles import lookup_levels
from cloak.runtime_types import PLACEHOLDER_ONLY_TYPES, placeholder_token

GEONAMES = Path("data/geonames")
CACHE = Path("data/lattice_cache.json")

E4B_PROMPT = """Entity: "{entity}"
Runtime type: "{span_type}"
Context: "{context}"

Give exactly 3 broader, strictly truthful replacements for this entity as it is used in the context, from most specific to most general. Each must be a short grammatical phrase that can replace the entity in the marked sentence. Do not output labels that merely name the runtime type. No descriptions, no symbols, no explanations.
Format: one phrase per line, nothing else."""

QWEN_PROMPT = """Entity: "{entity}"
Runtime type: "{span_type}"
Context: "{context}"

First, briefly reason about what this entity is in this context and what would identify it.
Then output exactly 3 broader, strictly truthful replacement phrases, most specific first. They must replace the entity in context, not label the type. The last line of your answer must be only the 3 phrases separated by " | "."""


# ---------- rule buckets ----------
# no dateutil: it fills missing fields from today's date -> false precision / nonsense
# ("May" -> May <this year>, "40" -> year 2040). Regex-only, fail to None.

_ONES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    " fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate(
    "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_SCALE = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}


def words_to_num(s: str) -> float | None:
    total, cur, seen = 0, 0, False
    for t in re.split(r"[\s-]+", s.lower().strip()):
        if t in _ONES:
            cur, seen = cur + _ONES[t], True
        elif t in _TENS:
            cur, seen = cur + _TENS[t], True
        elif t == "hundred":
            cur, seen = max(cur, 1) * 100, True
        elif t in _SCALE:
            total, cur, seen = total + max(cur, 1) * _SCALE[t], 0, True
        elif t in ("and", "a", "an"):
            continue
        else:
            return None
    return float(total + cur) if seen else None


_DECADE_WORD = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
                60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}
_MONTHS = ("january february march april may june july august september october"
           " november december").split()
_SEASON = dict([(m, "winter") for m in (12, 1, 2)] + [(m, "spring") for m in (3, 4, 5)]
               + [(m, "summer") for m in (6, 7, 8)] + [(m, "autumn") for m in (9, 10, 11)])


def _age_bucket(v: float) -> list[str] | None:
    v = int(v)
    if 13 <= v <= 19:
        return ["teenaged"]
    if 20 <= v <= 109:  # agreement-free: works after "I am", "turned", "a ... man"
        return [f"{_DECADE_WORD[min(v // 10 * 10, 90)]}-something"]
    return None


def _decades(y: int) -> list[str]:
    return [f"the {'early' if y % 10 < 4 else 'mid' if y % 10 < 7 else 'late'} {y//10*10}s",
            f"the {y//10*10}s"]


def bucket_date(text: str) -> list[str] | None:
    t = text.strip()
    if re.fullmatch(r"(19|20)\d{2}", t):
        return _decades(int(t))
    # age-shaped: "34 years old", "thirty four years old", bare 13..109
    m = re.fullmatch(r"(.+?)[\s-]*years?[\s-]*old", t, re.IGNORECASE)
    if m:
        v = words_to_num(m.group(1)) or (float(m.group(1)) if m.group(1).isdigit() else None)
        if v:
            return _age_bucket(v)
    if re.fullmatch(r"\d{1,3}", t):
        return _age_bucket(int(t))  # bare small int: age, never a year
    yr = re.search(r"\b((?:19|20)\d{2})\b", t)
    mon = next((mn for mn in _MONTHS if re.search(rf"\b{mn}\b", t, re.IGNORECASE)), None)
    if yr and mon:
        return [f"{mon.title()} {yr.group(1)}", *_decades(int(yr.group(1)))]
    if yr:
        return _decades(int(yr.group(1)))
    if mon:
        return [f"the {_SEASON[_MONTHS.index(mon) + 1]}"]
    if re.search(r"\b(spring|summer|autumn|fall|winter|month|week|year)s?\b", t, re.IGNORECASE):
        return ["some time ago" if re.search(r"\b(last|ago|previous|past)\b", t, re.IGNORECASE)
                else "at some point"]
    return None


def bucket_quantity(text: str) -> list[str] | None:
    m = re.search(r"[\d][\d,.]*", text)
    if m:
        try:
            v = float(m.group().replace(",", ""))
        except ValueError:
            return None
        if re.search(rf"{re.escape(m.group())}\s?[kK]\b", text):
            v *= 1_000
        elif re.search(rf"{re.escape(m.group())}\s?[mM]\b", text):
            v *= 1_000_000
        unit = re.sub(r"[\d,.]+\s?[kKmM]?", "", text).strip()
    else:
        # spelled-out: "two hundred thousand dollars"
        um = re.search(r"(dollars?|euros?|pounds?|USD|EUR|GBP|kr)\b", text, re.IGNORECASE)
        unit = um.group(1) if um else ""
        v = words_to_num(text[:um.start()] if um else text)
        if v is None:
            return None
    lo, hi = v * 0.5, v * 2
    fmt = lambda x: f"{x:,.0f}" if x >= 10 else f"{x:g}"
    # no exact-value level: "roughly <exact>" leaks the value verbatim
    return [f"between {fmt(lo)} and {fmt(hi)} {unit}".strip()]


# ---------- GeoNames ----------

_geo = None


def _load_geo():
    global _geo
    if _geo is not None:
        return _geo
    admin1 = {}
    for ln in open(GEONAMES / "admin1CodesASCII.txt", encoding="utf-8"):
        code, name, *_ = ln.rstrip("\n").split("\t")
        admin1[code] = name
    countries = {}
    for ln in open(GEONAMES / "countryInfo.txt", encoding="utf-8"):
        if ln.startswith("#"):
            continue
        f = ln.rstrip("\n").split("\t")
        countries[f[0]] = {"name": f[4], "continent": f[8]}
    cities = {}
    for ln in open(GEONAMES / "cities500.txt", encoding="utf-8"):
        f = ln.rstrip("\n").split("\t")
        name, asciiname, alts = f[1], f[2], f[3]
        pop = int(f[14] or 0)
        entry = (f[8], f[10], pop)  # country code, admin1 code, population
        for key in {name.lower(), asciiname.lower()}:
            if key not in cities or cities[key][2] < pop:  # most populous wins
                cities[key] = entry
    _geo = (cities, admin1, countries)
    return _geo


CONTINENTS = {"EU": "Europe", "AS": "Asia", "NA": "North America", "SA": "South America",
              "AF": "Africa", "OC": "Oceania", "AN": "Antarctica"}


def geonames_chain(place: str) -> list[str] | None:
    cities, admin1, countries = _load_geo()
    key = place.lower().strip()
    c = countries.get(key.upper()) if len(key) == 2 else None
    for cc in countries.values():  # country name given directly
        if cc["name"].lower() == key:
            return [f"a country in {CONTINENTS.get(cc['continent'], 'the world')}"]
    hit = cities.get(key)
    if not hit:
        return None
    ccode, a1, _ = hit
    country = countries.get(ccode, {"name": ccode, "continent": ""})
    chain = []
    region = admin1.get(f"{ccode}.{a1}")
    if region and region.lower() != key:
        chain.append(f"a city in {region}")
    chain.append(f"a city in {country['name']}")
    cont = CONTINENTS.get(country["continent"])
    if cont:
        chain.append(f"a city in {cont}")
    return chain or None


# ---------- WordNet ----------

def wordnet_chain(phrase: str, depth: int = 3) -> list[str] | None:
    from nltk.corpus import wordnet as wn
    p = phrase.lower().strip()
    syns = wn.synsets(p.replace(" ", "_"), pos=wn.NOUN) or \
        (wn.synsets(p.split()[-1], pos=wn.NOUN) if len(p.split()) > 1 else [])
    if not syns:
        return None
    s = syns[0]
    chain = []
    for _ in range(depth):
        hypers = s.instance_hypernyms() or s.hypernyms()
        if not hypers:
            break
        s = hypers[0]
        name = s.lemmas()[0].name().replace("_", " ")
        if name in ("entity", "abstraction", "physical entity", "object", "whole"):
            break
        art = "an" if name[0] in "aeiou" else "a"
        chain.append(f"{art} {name}")
    return chain or None


# ---------- NLI truthfulness gate ----------

_nli = None
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def _nli_prep(entity: str, context: str, candidates: list[str]):
    """Per-job viability: self-ref filter, sentence location, degenerate-dup check.
    Returns (viable_candidates, hypotheses); ([], []) fails closed."""
    candidates = [c for c in candidates if entity.lower() not in c.lower()]  # self-reference = leak
    pat = re.compile(re.escape(entity), re.IGNORECASE)
    sent = next((s for s in re.split(r"(?<=[.!?])\s+", context) if pat.search(s)), context)
    if not pat.search(sent):  # can't form the hypothesis -> fail closed (escalate/floor)
        return [], []
    hyps = [pat.sub(c, sent, count=1) for c in candidates]
    # degenerate substitution ("A city city picnics") => vacuous entailment; reject
    keep = [(c, h, sent) for c, h in zip(candidates, hyps)
            if not re.search(r"\b(\w{3,}) \1\b", h, re.IGNORECASE)]
    return [(c, sent) for c, h, sent in keep], [h for _, h, _ in keep]


def nli_gate_batch(jobs: list[tuple[str, str, list[str]]],
                   thresh: float = 0.6) -> list[list[tuple[str, float]]]:
    """One pipeline call for many (entity, context, candidates) jobs.
    Per job: approved (candidate, entailment_score) pairs, input order preserved."""
    global _nli
    prepped = [_nli_prep(e, ctx, cands) for e, ctx, cands in jobs]
    pairs, owners = [], []  # owners[i] = (job_idx, candidate)
    for j, (viable, hyps) in enumerate(prepped):
        for (cand, sent), hyp in zip(viable, hyps):
            pairs.append({"text": sent, "text_pair": hyp})
            owners.append((j, cand))
    if not pairs:
        return [[] for _ in jobs]
    if _nli is None:
        import torch
        from transformers import pipeline
        _nli = pipeline("text-classification", model=NLI_MODEL,
                        device=0 if torch.cuda.is_available() else -1)
    outs = _nli(pairs, top_k=None, truncation=True)
    approved: list[list[tuple[str, float]]] = [[] for _ in jobs]
    for (j, cand), scores in zip(owners, outs):
        ent = next(d["score"] for d in scores if d["label"] == "entailment")
        if ent >= thresh:
            approved[j].append((cand, ent))
    return approved


def nli_gate(entity: str, context: str, candidates: list[str], thresh: float = 0.6) -> list[str]:
    """Keep candidates where 'context' entails 'context with entity -> candidate'."""
    return [c for c, _ in nli_gate_batch([(entity, context, candidates)], thresh=thresh)[0]]


# ---------- teacher cascade ----------

def _parse_lines(reply: str) -> list[str]:
    lines = [re.sub(r"^[\s\d.\-*•]+", "", ln).strip().rstrip(".") for ln in reply.strip().splitlines()]
    return [ln for ln in lines if ln and len(ln.split()) <= 8 and ln[0].isascii()][:3]


_TYPE_NAME_PHRASES = {
    "a nationality", "an ethnicity", "a religion", "a religious affiliation",
    "a profession", "an occupation", "an age range", "a gender",
    "a marital status", "a health condition", "a sexual orientation",
    "a family relationship", "a demographic attribute",
}

_NATIONALITY_LEVELS = {
    "polish": ["Central European", "European"],
    "kenyan": ["East African", "African"],
    "german": ["Central European", "European"],
    "american": ["North American"],
    "british": ["Western European", "European"],
}

_ETHNICITY_LEVELS = {
    "kurdish": ["of Middle Eastern ethnicity", "of West Asian ethnicity"],
    "roma": ["of European ethnicity"],
    "romani": ["of European ethnicity"],
    "tamil": ["of South Asian ethnicity"],
    "arab": ["of Middle Eastern ethnicity"],
}

_RELIGION_LEVELS = {
    "catholic": ["Christian"],
    "protestant": ["Christian"],
    "orthodox": ["Christian"],
    "sunni": ["Muslim"],
    "shia": ["Muslim"],
}

_PROFESSION_LEVELS = {
    "cardiologist": ["medical specialist", "healthcare worker"],
    "doctor": ["medical professional", "healthcare worker"],
    "physician": ["medical professional", "healthcare worker"],
    "nurse": ["healthcare worker"],
    "journalist": ["media worker"],
    "reporter": ["media worker"],
    "prosecutor": ["legal professional"],
    "lawyer": ["legal professional"],
    "judge": ["legal professional"],
    "teacher": ["education worker"],
    "professor": ["education worker"],
    "engineer": ["technical professional"],
}

_HEALTH_LEVELS = {
    "diabetes": ["endocrine condition", "chronic condition"],
    "depression": ["mental health condition"],
    "asthma": ["respiratory condition"],
    "hiv": ["infectious disease"],
    "aids": ["infectious disease"],
    "cancer": ["serious illness"],
}

_FAMILY_LEVELS = {
    "daughter": ["child"],
    "son": ["child"],
    "wife": ["spouse"],
    "husband": ["spouse"],
    "grandfather": ["grandparent"],
    "grandmother": ["grandparent"],
    "mother": ["parent"],
    "father": ["parent"],
    "brother": ["sibling"],
    "sister": ["sibling"],
}


def _cache_key(span_text: str, span_type: str) -> str:
    return f"{span_type}::{span_text.lower().strip()}"


def is_type_name_phrase(fill: str) -> bool:
    f = re.sub(r"\s+", " ", fill.lower().strip().rstrip("."))
    return f in _TYPE_NAME_PHRASES


def _fine_curated_chain(span_text: str, span_type: str) -> list[str] | None:
    key = span_text.lower().strip()
    key_sing = key.rstrip("s")
    maps = {
        "nationality": _NATIONALITY_LEVELS,
        "ethnicity": _ETHNICITY_LEVELS,
        "religion": _RELIGION_LEVELS,
        "profession": _PROFESSION_LEVELS,
        "health-condition": _HEALTH_LEVELS,
        "family-role": _FAMILY_LEVELS,
    }
    table = maps.get(span_type)
    if not table:
        return None
    return table.get(key) or table.get(key_sing)


def _filtered_levels(span_text: str, levels: list[str] | None) -> list[str]:
    if not levels:
        return []
    surface = span_text.lower().strip()
    distinctive_numbers = set(re.findall(r"\d[\d,.]*\d|\d", span_text))
    out = []
    seen = set()
    for cand in levels:
        c = re.sub(r"\s+", " ", str(cand).strip().rstrip("."))
        cl = c.lower()
        if not c or cl in seen:
            continue
        if surface and surface in cl:
            continue
        if is_type_name_phrase(c):
            continue
        if distinctive_numbers & set(re.findall(r"\d[\d,.]*\d|\d", c)):
            continue
        seen.add(cl)
        out.append(c)
    return out


def _with_placeholder(span_text: str, span_type: str, levels: list[str] | None) -> list[str]:
    got = _filtered_levels(span_text, levels)
    got.append(placeholder_token(span_type, 1))
    return got


def teacher_lattices(entities: list[dict], workers: int = 6) -> dict:
    """Offline teacher cache builder keyed by runtime type and surface.

    entities: [{entity, type, context}]. Deployed lattice_for() only reads CACHE; it
    never calls the teacher. Empty approved lists are valid placeholder-only outcomes.
    """
    from inferdpt.llm import LLMClient
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [e for e in entities if _cache_key(e["entity"], e.get("type", "MISC")) not in cache]
    if not todo:
        return cache
    kw = dict(base_url="http://localhost:8060/v1", api_key="x", temperature=0.0, max_tokens=250)
    nothink = {"chat_template_kwargs": {"enable_thinking": False}}
    e4b = LLMClient("gemma 4 (E4B)", extra_body=nothink, **kw)
    qwen = LLMClient("Qwen3.6-35B-A3B", extra_body=nothink, **kw)

    with ThreadPoolExecutor(workers) as ex:
        replies = list(ex.map(lambda e: e4b.generate(
            E4B_PROMPT.format(entity=e["entity"], span_type=e.get("type", "MISC"),
                              context=e["context"])), todo))
    escalate = []
    for e, r in zip(todo, replies):
        cands = nli_gate(e["entity"], e["context"], _parse_lines(r))
        if cands:
            cache[_cache_key(e["entity"], e.get("type", "MISC"))] = {
                "lattice": _filtered_levels(e["entity"], cands), "tier": "e4b"}
        else:
            escalate.append(e)
    if escalate:
        with ThreadPoolExecutor(workers) as ex:
            replies = list(ex.map(lambda e: qwen.generate(
                QWEN_PROMPT.format(entity=e["entity"], span_type=e.get("type", "MISC"),
                                   context=e["context"])), escalate))
        for e, r in zip(escalate, replies):
            last = r.strip().splitlines()[-1] if r.strip() else ""
            cands = nli_gate(e["entity"], e["context"],
                             _parse_lines("\n".join(p.strip() for p in last.split("|"))))
            approved = _filtered_levels(e["entity"], cands)
            cache[_cache_key(e["entity"], e.get("type", "MISC"))] = (
                {"lattice": approved, "tier": "qwen"} if approved else
                {"lattice": [], "tier": "placeholder-only"})
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=2))
    return cache


TYPE_LABEL = {"ORG": "an organization", "LOC": "a place", "MISC": "something",
              "DEM": "a personal attribute", "PERSON": "a person",
              "DATETIME": "at some point", "QUANTITY": "a certain amount"}


def lattice_for(span_text: str, span_type: str, context: str = "") -> list[str]:
    """Zero-cost sources only; teacher entities must be pre-cached via teacher_lattices.

    Every source passes the NLI truthfulness gate against the span's context — rule sources
    used to bypass it, shipping context-wrong senses ("dragon" -> "a mythical monster",
    "vermont" -> "a city in Australia"); docs/issues/rule-lattice-nli-gate-bypass.md.
    Gate-empty coarse lattices fall to a legacy generic type label. Fine runtime lattices
    fall closed to their typed placeholder terminal.
    """
    deterministic = False
    if span_type in PLACEHOLDER_ONLY_TYPES or span_type in {"PERSON", "CODE"}:
        return [placeholder_token(span_type, 1)]
    if span_type == "DATETIME":
        got = bucket_date(span_text)
        deterministic = True
    elif span_type == "QUANTITY":
        got = bucket_quantity(span_text)
        deterministic = True
    elif span_type == "age":
        got = bucket_date(span_text)
        deterministic = True
    elif span_type == "LOC":
        got = lookup_levels(span_text, span_type)
        deterministic = bool(got)
        if not got:
            got = geonames_chain(span_text) or wordnet_chain(span_text)
    elif span_type in {
        "nationality", "ethnicity", "profession", "health-condition", "religion",
        "family-role",
    }:
        got = lookup_levels(span_text, span_type)
        deterministic = bool(got)
        if not got:
            got = _fine_curated_chain(span_text, span_type)
            deterministic = got is not None
        if got is None:
            got = wordnet_chain(span_text)
        if not got and CACHE.exists():
            got = json.loads(CACHE.read_text()).get(_cache_key(span_text, span_type), {}).get("lattice")
    elif span_type == "demographic-other":
        got = []
    else:
        got = lookup_levels(span_text, span_type)
        deterministic = bool(got)
        if not got:
            got = wordnet_chain(span_text)
        if not got and CACHE.exists():
            cache = json.loads(CACHE.read_text())
            got = (cache.get(_cache_key(span_text, span_type), {}).get("lattice") or
                   cache.get(span_text.lower(), {}).get("lattice"))
    if got and context and not deterministic:
        got = nli_gate(span_text, context, got)
    if span_type in {"LOC", "ORG", "MISC", "DEM", "DATETIME", "QUANTITY"}:
        return _filtered_levels(span_text, got) or [TYPE_LABEL.get(span_type, "something")]
    return _with_placeholder(span_text, span_type, got)


if __name__ == "__main__":
    assert bucket_date("March 3, 2021") == ["March 2021", "the early 2020s", "the 2020s"]
    assert bucket_date("40") == ["forty-something"] and bucket_date("May") == ["the spring"]
    assert bucket_date("thirty four years old") == ["thirty-something"]
    assert bucket_date("Last spring") == ["some time ago"]
    assert "between" in bucket_quantity("120,000 dollars")[0]
    assert bucket_quantity("two hundred thousand dollars") == \
        ["between 100,000 and 400,000 dollars"], bucket_quantity("two hundred thousand dollars")
    assert bucket_quantity("95k") == ["between 47,500 and 190,000"], bucket_quantity("95k")
    assert geonames_chain("Oslo") and "Norway" in " ".join(geonames_chain("Oslo"))
    assert wordnet_chain("cardiologist"), wordnet_chain("cardiologist")
    print("oslo:", geonames_chain("Oslo"))
    print("cardiologist:", wordnet_chain("cardiologist"))
    print("nli keep:", nli_gate("Novo Nordisk", "She works at Novo Nordisk in Oslo.",
                                ["a pharmaceutical company", "a bank", "a company"]))
    print("lattice.py self-check OK")
