import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from lattice_sources.common import ProfileRow, norm
from lattice_sources.occupation import _profession_levels

REGION_LEVELS = {
    "002": "african nationality",
    "003": "north american nationality",
    "005": "south american nationality",
    "009": "oceanian nationality",
    "013": "central american nationality",
    "014": "eastern african nationality",
    "015": "northern african nationality",
    "017": "middle african nationality",
    "018": "southern african nationality",
    "019": "american nationality",
    "021": "north american nationality",
    "029": "caribbean nationality",
    "030": "eastern asian nationality",
    "034": "southern asian nationality",
    "035": "southeast asian nationality",
    "039": "southern european nationality",
    "053": "oceanian nationality",
    "054": "oceanian nationality",
    "057": "oceanian nationality",
    "061": "oceanian nationality",
    "142": "asian nationality",
    "143": "central asian nationality",
    "145": "western asian nationality",
    "150": "european nationality",
    "151": "eastern european nationality",
    "154": "northern european nationality",
    "155": "western european nationality",
    "202": "african nationality",
    "419": "latin american nationality",
}

WIKIDATA_LEVELS = {
    "religion": {
        "buddhism": ["dharmic religion", "religious tradition"],
        "christianity": ["abrahamic religion", "religious tradition"],
        "hinduism": ["dharmic religion", "religious tradition"],
        "islam": ["abrahamic religion", "religious tradition"],
        "judaism": ["abrahamic religion", "religious tradition"],
        "sikhism": ["dharmic religion", "religious tradition"],
    },
    "nationality": {
        "france": ["western european nationality", "european nationality"],
        "germany": ["western european nationality", "european nationality"],
        "india": ["southern asian nationality", "asian nationality"],
        "italy": ["southern european nationality", "european nationality"],
        "united kingdom": ["northern european nationality", "european nationality"],
        "united states": ["north american nationality", "american nationality"],
        "united states of america": ["north american nationality", "american nationality"],
    },
}

DEMONYM_ALIASES = {
    "AR": ["argentine", "argentinian"],
    "AU": ["australian"],
    "BR": ["brazilian"],
    "CA": ["canadian"],
    "CN": ["chinese"],
    "DE": ["german"],
    "ES": ["spanish"],
    "FR": ["french"],
    "GB": ["british"],
    "IN": ["indian"],
    "IT": ["italian"],
    "JP": ["japanese"],
    "MX": ["mexican"],
    "NL": ["dutch"],
    "RU": ["russian"],
    "US": ["american"],
}


def rows_from_cldr_zip(path: Path) -> list[ProfileRow]:
    with zipfile.ZipFile(path) as zf:
        territories = json.loads(zf.read("cldr-localenames-full/main/en/territories.json").decode("utf-8"))
        containment = json.loads(zf.read("cldr-core/supplemental/territoryContainment.json").decode("utf-8"))
    return rows_from_cldr_territories(
        territories,
        containment["supplemental"]["territoryContainment"],
    )


def rows_from_cldr_territories(path_or_obj, containment: dict) -> list[ProfileRow]:
    if isinstance(path_or_obj, (str, Path)):
        data = json.loads(Path(path_or_obj).read_text())
    else:
        data = path_or_obj
    territories = data["main"]["en"]["localeDisplayNames"]["territories"]
    aliases: dict[str, list[str]] = {}
    for code, name in territories.items():
        if "-alt-" in code:
            base = code.split("-alt-", 1)[0]
            aliases.setdefault(base, []).append(norm(name))

    parent = {}
    for region, spec in containment.items():
        if "status" in region:
            continue
        for child in spec.get("_contains", []):
            parent.setdefault(child, region)

    rows = []
    for code, name in territories.items():
        if not code.isalpha() or len(code) != 2:
            continue
        surface = norm(name)
        levels = _nationality_levels(code, parent)
        if not surface or not levels:
            continue
        rows.append(ProfileRow(
            runtime_type="nationality",
            surface=surface,
            aliases=sorted({
                a
                for a in [*aliases.get(code, []), *DEMONYM_ALIASES.get(code, [])]
                if a and a != surface
            }),
            levels=levels,
            source_ids=[f"cldr:{code}"],
        ))
    return rows


def _nationality_levels(code: str, parent: dict[str, str]) -> list[str]:
    cur = parent.get(code)
    seen = set()
    levels = []
    while cur and cur not in seen:
        seen.add(cur)
        if cur in REGION_LEVELS:
            level = REGION_LEVELS[cur]
            if level not in levels:
                levels.append(level)
        cur = parent.get(cur)
    return levels


def rows_from_wikidata_sparql_xml(path: Path) -> list[ProfileRow]:
    ns = {"s": "http://www.w3.org/2005/sparql-results#"}
    groups = {}
    root = ET.parse(path).getroot()
    for result in root.findall(".//s:result", ns):
        runtime_type = _binding_text(result, "type", ns)
        item = _binding_uri_tail(result, "item", ns)
        label = norm(_binding_text(result, "itemLabel", ns))
        alias = norm(_binding_text(result, "alias", ns))
        if runtime_type not in {"profession", "religion", "nationality"} or not label:
            continue
        key = (runtime_type, item, label)
        groups.setdefault(key, set())
        if alias and alias != label:
            groups[key].add(alias)

    rows = []
    for (runtime_type, item, label), aliases in sorted(groups.items()):
        rows.append(ProfileRow(
            runtime_type=runtime_type,
            surface=label,
            aliases=sorted(aliases),
            levels=_wikidata_levels(runtime_type, label),
            source_ids=[f"wikidata:{item}"],
        ))
    return [r for r in rows if r.levels]


def _binding_text(result: ET.Element, name: str, ns: dict[str, str]) -> str:
    node = result.find(f"s:binding[@name='{name}']/s:literal", ns)
    return node.text if node is not None and node.text else ""


def _binding_uri_tail(result: ET.Element, name: str, ns: dict[str, str]) -> str:
    node = result.find(f"s:binding[@name='{name}']/s:uri", ns)
    return node.text.rsplit("/", 1)[-1] if node is not None and node.text else ""


def _wikidata_levels(runtime_type: str, label: str) -> list[str]:
    if runtime_type == "profession":
        return _profession_levels(label)
    return WIKIDATA_LEVELS.get(runtime_type, {}).get(label, [f"{runtime_type} group"])
