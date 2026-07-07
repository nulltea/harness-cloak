import re
from pathlib import Path

from lattice_sources.common import ProfileRow, norm

CONTINENTS = {
    "AF": "africa",
    "AN": "antarctica",
    "AS": "asia",
    "EU": "europe",
    "NA": "north america",
    "OC": "oceania",
    "SA": "south america",
}


def rows_from_geonames(geo_dir: Path) -> list[ProfileRow]:
    geo_dir = Path(geo_dir)
    if not (geo_dir / "countryInfo.txt").exists():
        return []
    countries = _countries(geo_dir / "countryInfo.txt")
    rows = _country_rows(countries)
    if (geo_dir / "cities500.txt").exists():
        admin1 = _admin1(geo_dir / "admin1CodesASCII.txt")
        rows.extend(_city_rows(geo_dir / "cities500.txt", countries, admin1))
    return rows


def _countries(path: Path) -> dict[str, dict]:
    countries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 17:
            continue
        iso2, iso3, _, fips, name, capital, _, population, continent = f[:9]
        countries[iso2] = {
            "iso2": iso2,
            "iso3": iso3,
            "fips": fips,
            "name": name,
            "capital": capital,
            "population": int(population or 0),
            "continent": continent,
        }
    return countries


def _admin1(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        f = line.rstrip("\n").split("\t")
        if len(f) >= 2:
            out[f[0]] = f[1]
    return out


def _country_rows(countries: dict[str, dict]) -> list[ProfileRow]:
    rows = []
    for country in countries.values():
        surface = norm(country["name"])
        continent = CONTINENTS.get(country["continent"], "the world")
        level = f"a country in {continent}"
        if surface and surface in norm(level):
            level = "a geographic area"
        aliases = [country["iso2"], country["iso3"], country["fips"]]
        rows.append(ProfileRow(
            runtime_type="LOC",
            surface=surface,
            aliases=[norm(a) for a in aliases if norm(a) and norm(a) != surface],
            levels=[level],
            source_ids=[f"geonames-country:{country['iso2']}"],
            count=max(float(country["population"]), 1.0),
        ))
    return rows


def _city_rows(path: Path, countries: dict[str, dict], admin1: dict[str, str]) -> list[ProfileRow]:
    rows_by_surface = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        f = line.rstrip("\n").split("\t")
        if len(f) < 19:
            continue
        geoname_id, name, ascii_name, alt_names = f[0], f[1], f[2], f[3]
        country_code, admin1_code = f[8], f[10]
        population = int(f[14] or 0)
        country = countries.get(country_code)
        if not country:
            continue
        surface = norm(name)
        levels = []
        region = norm(admin1.get(f"{country_code}.{admin1_code}", ""))
        for level in (
            f"a city in {region}" if region else "",
            f"a city in {norm(country['name'])}",
            f"a city in {CONTINENTS.get(country['continent'], 'the world')}",
        ):
            level = norm(level)
            if level and surface not in level and level not in levels:
                levels.append(level)
        if not levels:
            continue
        aliases = {norm(ascii_name)}
        aliases.update(_reviewable_alias(a) for a in alt_names.split(","))
        aliases.discard(surface)
        aliases.discard("")
        row = ProfileRow(
            runtime_type="LOC",
            surface=surface,
            aliases=sorted(aliases),
            levels=levels,
            source_ids=[f"geonames:{geoname_id}"],
            count=max(float(population), 1.0),
        )
        cur = rows_by_surface.get(surface)
        if cur is None or row.count > cur.count:
            rows_by_surface[surface] = row
    return list(rows_by_surface.values())


def _reviewable_alias(alias: str) -> str:
    alias = norm(alias)
    if not alias or len(alias) > 48:
        return ""
    if not re.fullmatch(r"[a-z0-9 .,'()&/-]+", alias):
        return ""
    return alias
