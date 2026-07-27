"""Deterministically ground level_counts/level_grounding for LOC, nationality, and
organization-medical-facility rows in data/lattice_profiles/lattice_profiles.json.

Unlike the other dataset-seeded types (whose per-level anonymity sizes are not grounded, so we
leave them count-only), these three have levels drawn from taxonomies whose sizes are
deterministically countable:

  LOC         -- from data/geonames/ (the same files scripts/lattice_sources/geonames.py builds
                 the levels from): "a city in <country>" = #cities in that country in cities500,
                 "a city in <continent>" = #cities in that continent, "a country in <continent>"
                 = #countries in countryInfo, "a city in <admin1>" = #cities in that admin1.
                 Rows are disambiguated by source_id (geonames:<id> / geonames-country:<iso2>)
                 back to exact GeoNames codes -- level strings alone are ambiguous ("a city in
                 georgia" is both the country and the US state). Real-world, status="certifying".

  nationality -- from CLDR M49 territoryContainment (the same tree demographics.rows_from_cldr
                 walks): each region level's count = #leaf countries under the M49 node(s) that
                 map to that level string. Monotone by nesting. Real-world, status="certifying".

  organization-medical-facility
              -- corpus-membership (user decision): count = #entries in THIS profile whose levels
                 include the level string. We lack the full NPPES universe locally (only a weekly
                 increment ships in data/lattice_sources/raw/org/), and its taxonomy-prefix->level
                 mapping would need the full monthly file to count a real anonymity set; the weekly
                 increment is not a valid universe. Corpus-membership is a conservative undercount
                 (understates anonymity -- safe for a k-anonymity walk), so it is honest but not
                 certifying: status="model-proposed", count_basis="corpus-membership". Its chains
                 mix sibling taxonomies (e.g. home-health + transportation, both under healthcare
                 organization), so levels are reordered by count to stay monotone.

Real-world counts (LOC/nationality) are magnitudes from certifying sources, not fabricated. A row
whose counts come out non-monotone along its chain (a data-quality error, e.g. a LOC entry that
conflates a country and a same-named city) is left count-only rather than repaired -- we never
fudge a k-anonymity walk. Run with --write to apply.
"""
from __future__ import annotations

import collections
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from lattice_sources.common import norm  # noqa: E402
from lattice_sources.geonames import CONTINENTS  # noqa: E402
from lattice_sources.demographics import REGION_LEVELS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cloak.lattice.profiles import validate_profile_artifact  # noqa: E402

ART = Path("data/lattice_profiles/lattice_profiles.json")
GEO = Path("data/geonames")
CLDR = Path("data/lattice_sources/raw/nationality/cldr-48.2.0-json-full.zip")


# ---- GeoNames (LOC) --------------------------------------------------------

def _load_countries() -> dict[str, dict]:
    countries = {}
    for line in (GEO / "countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9 or not f[0]:
            continue
        countries[f[0]] = {"name": f[4], "continent": f[8]}
    return countries


def _load_geo_universe(countries: dict[str, dict], needed_ids: set[str]) -> dict:
    admin1 = {}
    for line in (GEO / "admin1CodesASCII.txt").read_text(encoding="utf-8").splitlines():
        f = line.rstrip("\n").split("\t")
        if len(f) >= 2:
            admin1[f[0]] = f[1]
    cont_city, country_city, admin1_city = collections.Counter(), collections.Counter(), collections.Counter()
    id_codes: dict[str, tuple[str, str]] = {}
    with open(GEO / "cities500.txt", encoding="utf-8") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 19:
                continue
            gid, cc, a1 = f[0], f[8], f[10]
            country = countries.get(cc)
            if not country:
                continue
            cont_city[country["continent"]] += 1
            country_city[cc] += 1
            admin1_city[f"{cc}.{a1}"] += 1
            if gid in needed_ids:
                id_codes[gid] = (cc, a1)
    cont_country = collections.Counter(c["continent"] for c in countries.values())
    return {
        "admin1": admin1, "cont_city": cont_city, "country_city": country_city,
        "admin1_city": admin1_city, "id_codes": id_codes, "cont_country": cont_country,
    }


def _loc_level_counts(row: dict, countries: dict, geo: dict) -> tuple[dict, dict]:
    """Reconstruct each level's exact GeoNames string (mirroring geonames.py) so we assign the
    right universe count to the right level, then key counts/grounding by the row's actual level
    text."""
    sids = row.get("source_ids", [])
    country_sid = next((s for s in sids if s.startswith("geonames-country:")), None)
    city_sid = next((s for s in sids if s.startswith("geonames:")), None)
    expected: dict[str, tuple[float, dict]] = {}  # norm(level) -> (count, grounding)
    if country_sid:
        iso2 = country_sid.split(":", 1)[1].upper()
        c = countries.get(iso2)
        if c:
            cont = CONTINENTS.get(c["continent"], "the world")
            expected[norm(f"a country in {cont}")] = (
                float(geo["cont_country"][c["continent"]]),
                {"status": "certifying", "source_family": "geonames-universe",
                 "selector": f"countryInfo.continent == {c['continent']}",
                 "member_set_ref": f"geonames:countries:{c['continent']}"},
            )
    elif city_sid:
        codes = geo["id_codes"].get(city_sid.split(":", 1)[1])
        if codes:
            cc, a1 = codes
            c = countries[cc]
            cont = CONTINENTS.get(c["continent"], "the world")
            region = norm(geo["admin1"].get(f"{cc}.{a1}", ""))
            if region:
                expected[norm(f"a city in {region}")] = (
                    float(geo["admin1_city"][f"{cc}.{a1}"]),
                    {"status": "certifying", "source_family": "geonames-universe",
                     "selector": f"cities500.admin1 == {cc}.{a1}",
                     "member_set_ref": f"geonames:cities500:{cc}.{a1}"},
                )
            expected[norm(f"a city in {norm(c['name'])}")] = (
                float(geo["country_city"][cc]),
                {"status": "certifying", "source_family": "geonames-universe",
                 "selector": f"cities500.country_code == {cc}",
                 "member_set_ref": f"geonames:cities500:{cc}"},
            )
            expected[norm(f"a city in {cont}")] = (
                float(geo["cont_city"][c["continent"]]),
                {"status": "certifying", "source_family": "geonames-universe",
                 "selector": f"cities500.continent == {c['continent']}",
                 "member_set_ref": f"geonames:cities500:continent:{c['continent']}"},
            )
    lc, lg = {}, {}
    for level in row.get("levels", []):
        hit = expected.get(norm(level))
        if hit:
            lc[level], lg[level] = hit
    return lc, lg


# ---- CLDR M49 (nationality) -----------------------------------------------

def _load_m49_contains() -> dict[str, list[str]]:
    with zipfile.ZipFile(CLDR) as zf:
        containment = json.loads(zf.read("cldr-core/supplemental/territoryContainment.json").decode("utf-8"))
    contains = {}
    for region, spec in containment["supplemental"]["territoryContainment"].items():
        if "-status-" in region:
            continue
        contains[region] = spec.get("_contains", [])
    return contains


def _alpha2_descendants(region: str, contains: dict, memo: dict) -> set[str]:
    if region in memo:
        return memo[region]
    out: set[str] = set()
    for child in contains.get(region, []):
        if child.isalpha() and len(child) == 2:
            out.add(child)
        else:
            out |= _alpha2_descendants(child, contains, memo)
    memo[region] = out
    return out


def _level_string_universe(contains: dict) -> dict[str, int]:
    """count of leaf countries per nationality level string, unioning the M49 codes that map to
    that string (e.g. 002+202 -> 'african nationality')."""
    memo: dict[str, set[str]] = {}
    by_string: dict[str, set[str]] = collections.defaultdict(set)
    for code, level in REGION_LEVELS.items():
        by_string[level] |= _alpha2_descendants(code, contains, memo)
    counts = {level: len(members) for level, members in by_string.items()}
    counts["nationality group"] = len(_alpha2_descendants("001", contains, memo))
    return counts


def _nationality_level_counts(row: dict, universe: dict[str, int]) -> tuple[dict, dict]:
    lc, lg = {}, {}
    for level in row.get("levels", []):
        n = universe.get(norm(level))
        if n:
            lc[level] = float(n)
            lg[level] = {"status": "certifying", "source_family": "cldr-m49",
                         "selector": f"CLDR M49 territoryContainment: countries under '{level}'",
                         "member_set_ref": f"cldr-m49:{norm(level)}"}
    return lc, lg


# ---- organization-medical-facility (corpus-membership) --------------------

def _org_corpus_counts(org_entries: dict) -> dict[str, int]:
    counts: collections.Counter = collections.Counter()
    for row in org_entries.values():
        for level in row.get("levels", []):
            counts[norm(level)] += 1
    return counts


def _org_level_counts(row: dict, counts: dict[str, int]) -> tuple[list[str], dict, dict]:
    """corpus-membership counts. levels are reordered by count so the k-anonymity walk is monotone:
    a row's sibling taxonomies (e.g. home-health + transportation, both under 'healthcare
    organization') carry no intrinsic narrow->broad order, and 'healthcare organization' (largest)
    must land last. Returns (reordered_levels, level_counts, level_grounding)."""
    levels = sorted(row.get("levels", []), key=lambda level: counts.get(norm(level), 0))
    lc, lg = {}, {}
    for level in levels:
        n = counts.get(norm(level))
        if n:
            lc[level] = float(n)
            lg[level] = {
                "status": "model-proposed", "source_family": "corpus-membership",
                "count_basis": "corpus-membership",
                "count_evidence": (f"'{level}' count is a corpus-membership anonymity-set size "
                                   "(number of organization-medical-facility entries in this profile "
                                   "whose levels include it); conservative undercount of the real "
                                   "NPPES universe, not certifying"),
                "selector": f"count of organization-medical-facility entries generalizing to '{level}'",
                "member_set_ref": None,
            }
    return levels, lc, lg


# ---- driver ----------------------------------------------------------------

def _monotone(levels: list[str], lc: dict) -> bool:
    covered = [lc[l] for l in levels if l in lc]
    return all(a <= b for a, b in zip(covered, covered[1:]))


def main() -> None:
    write = "--write" in sys.argv
    art = json.loads(ART.read_text())
    countries = _load_countries()

    loc = art["profiles"]["LOC"]
    needed_ids = {s.split(":", 1)[1] for row in loc.values() for s in row.get("source_ids", []) if s.startswith("geonames:")}
    geo = _load_geo_universe(countries, needed_ids)
    contains = _load_m49_contains()
    universe = _level_string_universe(contains)

    stats = collections.Counter()
    non_monotone = []
    for rt, compute in (("LOC", lambda r: _loc_level_counts(r, countries, geo)),
                        ("nationality", lambda r: _nationality_level_counts(r, universe))):
        for surface, row in art["profiles"][rt].items():
            lc, lg = compute(row)
            stats[f"{rt}:rows"] += 1
            if not lc:
                stats[f"{rt}:no-count"] += 1
                continue
            if not _monotone(row["levels"], lc):
                non_monotone.append((rt, surface, [(l, lc.get(l)) for l in row["levels"]]))
                stats[f"{rt}:non-monotone-skipped"] += 1
                continue
            stats[f"{rt}:grounded"] += 1
            missing = sum(1 for l in row["levels"] if l not in lc)
            if missing:
                stats[f"{rt}:partial({missing} lvl uncounted)"] += 1
            if write:
                row["level_counts"] = lc
                row["level_grounding"] = lg

    org = art["profiles"].get("organization-medical-facility", {})
    org_counts = _org_corpus_counts(org)
    for surface, row in org.items():
        levels, lc, lg = _org_level_counts(row, org_counts)
        stats["org:rows"] += 1
        if not lc:
            stats["org:no-count"] += 1
            continue
        stats["org:grounded"] += 1
        if write:
            row["levels"] = levels
            row["level_counts"] = lc
            row["level_grounding"] = lg

    print("stats:", dict(stats))
    print(f"non-monotone (left count-only): {len(non_monotone)}")
    for rt, s, chain in non_monotone[:20]:
        print(f"  {rt}:{s}: {chain}")

    if write:
        errs = validate_profile_artifact(art)
        if errs:
            raise SystemExit("validation failed:\n" + "\n".join(errs[:30]))
        ART.write_text(json.dumps(art, indent=2, sort_keys=True))
        print(f"wrote {ART} (validation OK)")
    else:
        print("dry run -- pass --write to apply")


if __name__ == "__main__":
    main()
