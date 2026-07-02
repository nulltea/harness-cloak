"""Generalization lattices: rule buckets, GeoNames chains, WordNet paths,
and the E4B->Qwen teacher cascade with an NLI truthfulness gate.

A lattice is an ordered list of surface phrases, most specific -> most general.
Plan: docs/plans/2026-07-02-d1-prototype-implementation.md P1.2.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GEONAMES = Path("data/geonames")
CACHE = Path("data/lattice_cache.json")

E4B_PROMPT = """Entity: "{entity}"
Context: "{context}"

Give exactly 3 broader, strictly truthful generalizations of this entity as it is used in the context, from most specific to most general. Each must be a short English noun phrase (with article) that the entity IS an instance of. No descriptions, no symbols, no explanations.
Format: one phrase per line, nothing else."""

QWEN_PROMPT = """Entity: "{entity}"
Context: "{context}"

First, briefly reason about what this entity is in this context and what would identify it.
Then output exactly 3 broader, strictly truthful generalizations, most specific first: short noun phrases (with article) such that the entity IS an instance of each. The last line of your answer must be only the 3 phrases separated by " | "."""


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


def nli_gate(entity: str, context: str, candidates: list[str], thresh: float = 0.6) -> list[str]:
    """Keep candidates where 'context' entails 'context with entity -> candidate'."""
    global _nli
    if _nli is None:
        import torch
        from transformers import pipeline
        _nli = pipeline("text-classification", model=NLI_MODEL,
                        device=0 if torch.cuda.is_available() else -1)
    candidates = [c for c in candidates if entity.lower() not in c.lower()]  # self-reference = leak
    pat = re.compile(re.escape(entity), re.IGNORECASE)
    sent = next((s for s in re.split(r"(?<=[.!?])\s+", context) if pat.search(s)), context)
    if not pat.search(sent):  # can't form the hypothesis -> fail closed (escalate/floor)
        return []
    hyps = [pat.sub(c, sent, count=1) for c in candidates]
    # degenerate substitution ("A city city picnics") => vacuous entailment; reject
    ok = [not re.search(r"\b(\w{3,}) \1\b", h, re.IGNORECASE) for h in hyps]
    pairs = [{"text": sent, "text_pair": h} for h, o in zip(hyps, ok) if o]
    outs = _nli(pairs, top_k=None, truncation=True) if pairs else []
    keep = []
    for c, scores in zip([c for c, o in zip(candidates, ok) if o], outs):
        ent_score = next(d["score"] for d in scores if d["label"] == "entailment")
        if ent_score >= thresh:
            keep.append(c)
    return keep


# ---------- teacher cascade ----------

def _parse_lines(reply: str) -> list[str]:
    lines = [re.sub(r"^[\s\d.\-*•]+", "", ln).strip().rstrip(".") for ln in reply.strip().splitlines()]
    return [ln for ln in lines if ln and len(ln.split()) <= 8 and ln[0].isascii()][:3]


def teacher_lattices(entities: list[dict], workers: int = 6) -> dict:
    """entities: [{entity, context}]. Returns {entity: {lattice, tier}}; caches to CACHE.

    Cascade: E4B (parallel, rationale-free) -> NLI gate -> Qwen3.6 CoT retry -> type-label floor.
    """
    from inferdpt.llm import LLMClient
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [e for e in entities if e["entity"] not in cache]
    if not todo:
        return cache
    kw = dict(base_url="http://localhost:8060/v1", api_key="x", temperature=0.0, max_tokens=250)
    nothink = {"chat_template_kwargs": {"enable_thinking": False}}
    e4b = LLMClient("gemma 4 (E4B)", extra_body=nothink, **kw)
    qwen = LLMClient("Qwen3.6-35B-A3B", extra_body=nothink, **kw)

    with ThreadPoolExecutor(workers) as ex:
        replies = list(ex.map(lambda e: e4b.generate(
            E4B_PROMPT.format(entity=e["entity"], context=e["context"])), todo))
    escalate = []
    for e, r in zip(todo, replies):
        cands = nli_gate(e["entity"], e["context"], _parse_lines(r))
        if cands:
            cache[e["entity"]] = {"lattice": cands, "tier": "e4b"}
        else:
            escalate.append(e)
    if escalate:
        with ThreadPoolExecutor(workers) as ex:
            replies = list(ex.map(lambda e: qwen.generate(
                QWEN_PROMPT.format(entity=e["entity"], context=e["context"])), escalate))
        for e, r in zip(escalate, replies):
            last = r.strip().splitlines()[-1] if r.strip() else ""
            cands = nli_gate(e["entity"], e["context"],
                             _parse_lines("\n".join(p.strip() for p in last.split("|"))))
            cache[e["entity"]] = ({"lattice": cands, "tier": "qwen"} if cands else
                                  {"lattice": [], "tier": "floor"})
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=2))
    return cache


TYPE_LABEL = {"ORG": "an organization", "LOC": "a place", "MISC": "something",
              "DEM": "a personal attribute", "PERSON": "a person",
              "DATETIME": "at some point", "QUANTITY": "a certain amount"}


def lattice_for(span_text: str, span_type: str, context: str = "") -> list[str]:
    """Zero-cost sources only; teacher entities must be pre-cached via teacher_lattices."""
    if span_type == "DATETIME":
        got = bucket_date(span_text)
    elif span_type == "QUANTITY":
        got = bucket_quantity(span_text)
    elif span_type == "LOC":
        got = geonames_chain(span_text) or wordnet_chain(span_text)
    else:
        got = wordnet_chain(span_text)
        if not got and CACHE.exists():
            got = json.loads(CACHE.read_text()).get(span_text.lower(), {}).get("lattice")
    return got or [TYPE_LABEL.get(span_type, "something")]


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
