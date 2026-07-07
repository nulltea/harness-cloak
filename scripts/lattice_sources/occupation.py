import csv
import xml.etree.ElementTree as ET
from pathlib import Path

from lattice_sources.common import ProfileRow, norm

SOC_MAJOR_LEVELS = {
    "11": ["management occupation", "professional worker"],
    "13": ["business and financial occupation", "professional worker"],
    "15": ["computer and mathematical occupation", "professional worker"],
    "17": ["architecture and engineering occupation", "professional worker"],
    "19": ["science occupation", "professional worker"],
    "21": ["community and social service occupation", "professional worker"],
    "23": ["legal professional", "professional worker"],
    "25": ["education worker", "professional worker"],
    "27": ["arts and media worker", "professional worker"],
    "29": ["healthcare worker"],
    "31": ["healthcare support worker", "healthcare worker"],
    "33": ["protective service worker"],
    "35": ["food service worker"],
    "37": ["building and grounds worker"],
    "39": ["personal care and service worker"],
    "41": ["sales worker"],
    "43": ["clerical worker"],
    "45": ["agricultural worker"],
    "47": ["construction worker"],
    "49": ["installation and repair worker"],
    "51": ["production worker"],
    "53": ["transportation and material moving worker"],
    "55": ["military occupation"],
}

ISCO_MAJOR_LEVELS = {
    "1": ["management occupation"],
    "2": ["professional worker"],
    "3": ["technical worker"],
    "4": ["clerical worker"],
    "5": ["service and sales worker"],
    "6": ["agricultural worker"],
    "7": ["craft and trades worker"],
    "8": ["machine and plant operator"],
    "9": ["elementary occupation worker"],
}


def _profession_levels(title: str, major_group: str = "", code: str = "") -> list[str]:
    t = norm(f"{title} {major_group}")
    if "professional" in norm(major_group):
        return ["professional worker"]
    if any(w in t for w in ("journalist", "reporter", "news analyst", "correspondent")):
        return ["media worker", "professional worker"]
    if any(w in t for w in ("medical", "physician", "doctor", "nurse", "health")):
        return ["healthcare worker"]
    if any(w in t for w in ("law", "legal", "judge", "prosecutor", "attorney")):
        return ["legal professional", "professional worker"]
    if any(w in t for w in ("teacher", "education", "professor", "school")):
        return ["education worker", "professional worker"]
    code_levels = _code_levels(code)
    if code_levels:
        return code_levels
    if "professional" in t or major_group:
        return ["professional worker"]
    return ["worker"]


def _code_levels(code: str) -> list[str]:
    code = code.strip().upper()
    if len(code) >= 2 and code[:2].isdigit() and code[:2] in SOC_MAJOR_LEVELS:
        return SOC_MAJOR_LEVELS[code[:2]]
    if code.startswith("C") and len(code) >= 2 and code[1] in ISCO_MAJOR_LEVELS:
        return ISCO_MAJOR_LEVELS[code[1]]
    if code and code[0] in ISCO_MAJOR_LEVELS:
        return ISCO_MAJOR_LEVELS[code[0]]
    return []


def rows_from_onet_titles(path: Path) -> list[ProfileRow]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            title = norm(r.get("Title", ""))
            alt = norm(r.get("Alternate Title", ""))
            if not title or not alt:
                continue
            rows.append(ProfileRow(
                runtime_type="profession",
                surface=title,
                aliases=[alt],
                levels=_profession_levels(f"{title} {alt}", code=r.get("O*NET-SOC Code", "").strip()),
                source_ids=[f"onet:{r.get('O*NET-SOC Code', '').strip()}"],
            ))
    return rows


def rows_from_onet_job_titles(path: Path) -> list[ProfileRow]:
    rows = []
    records = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            title = norm(r.get("Job Title", ""))
            if not title:
                continue
            code = r.get("O*NET-SOC Code", "").strip()
            records.append((title, code))
    majors_by_title = {}
    for title, code in records:
        if len(code) >= 2:
            majors_by_title.setdefault(title, set()).add(code[:2])
    for title, code in records:
        if len(majors_by_title.get(title, set())) > 1:
            continue
        rows.append(ProfileRow(
            runtime_type="profession",
            surface=title,
            aliases=[],
            levels=_profession_levels(title, code=code),
            source_ids=[f"onet-job-title:{code}"],
        ))
    return rows


def rows_from_isco_csv(path: Path) -> list[ProfileRow]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            title = norm(r.get("title", ""))
            if not title:
                continue
            rows.append(ProfileRow(
                runtime_type="profession",
                surface=title,
                aliases=[],
                levels=_profession_levels(title, r.get("major_group", "")),
                source_ids=[f"isco:{r.get('code', '').strip()}"],
            ))
    return rows


def rows_from_esco_rdf(path_or_file) -> list[ProfileRow]:
    ns = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "skos": "http://www.w3.org/2004/02/skos/core#",
    }
    rows = []
    context = ET.iterparse(path_or_file, events=("end",))
    for _, elem in context:
        if elem.tag != f"{{{ns['skos']}}}Concept":
            continue
        about = elem.attrib.get(f"{{{ns['rdf']}}}about", "")
        if "/esco/occupation/" not in about:
            elem.clear()
            continue
        pref = _direct_lang_texts(elem, f"{{{ns['skos']}}}prefLabel", "en")
        if not pref:
            elem.clear()
            continue
        surface = norm(pref[0])
        aliases = [
            a for a in _direct_lang_texts(elem, f"{{{ns['skos']}}}altLabel", "en")
            if norm(a) and norm(a) != surface
        ]
        broader_labels = [
            t.text or ""
            for broader in elem.findall(f"{{{ns['skos']}}}broader")
            for t in broader.iter(f"{{{ns['skos']}}}prefLabel")
            if t.attrib.get("{http://www.w3.org/XML/1998/namespace}lang") == "en"
        ]
        broader_codes = [
            code
            for broader in elem.findall(f"{{{ns['skos']}}}broader")
            for code in [_resource_code(broader.attrib.get(f"{{{ns['rdf']}}}resource", ""))]
            if code
        ]
        rows.append(ProfileRow(
            runtime_type="profession",
            surface=surface,
            aliases=sorted({norm(a) for a in aliases}),
            levels=_profession_levels(" ".join([surface, *broader_labels]), code=broader_codes[0] if broader_codes else ""),
            source_ids=[f"esco:{about.rsplit('/', 1)[-1]}"],
        ))
        elem.clear()
    return rows


def _resource_code(resource: str) -> str:
    tail = resource.rsplit("/", 1)[-1].strip()
    if tail.startswith("C") and len(tail) >= 2 and tail[1].isdigit():
        return tail
    return ""


def _direct_lang_texts(elem: ET.Element, tag: str, lang: str) -> list[str]:
    out = []
    for child in list(elem):
        if child.tag == tag and child.attrib.get("{http://www.w3.org/XML/1998/namespace}lang") == lang:
            text = norm(child.text or "")
            if text:
                out.append(text)
    return out
