import json
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from lattice_sources.categorical import alias_rows
from lattice_sources.common import ProfileRow
from lattice_sources.demographics import rows_from_cldr_territories, rows_from_wikidata_sparql_xml
from lattice_sources.geonames import rows_from_geonames
from lattice_sources.legacy_cache import rows_from_legacy_teacher_cache
from lattice_sources.obo import rows_from_obo
from lattice_sources.occupation import rows_from_esco_rdf, rows_from_isco_csv, rows_from_onet_job_titles, rows_from_onet_titles
from lattice_sources.religion import rows_from_arda_variable_labels
from lattice_sources.drugs import rows_from_openfda_ndc_zip
from lattice_sources.organizations import rows_from_nppes_zip
from lattice_sources.procedures import rows_from_icd10_pcs_order_zip
from build_lattice_profiles import collect_rows, coverage_report, merge_rows
from download.fetch_lattice_sources import fetch, source_report


def test_onet_titles_to_profession_rows(tmp_path):
    src = tmp_path / "Alternate Titles.txt"
    src.write_text(
        "O*NET-SOC Code\tTitle\tAlternate Title\tShort Title\tSource(s)\n"
        "27-3023.00\tNews Analysts, Reporters, and Journalists\tReporter\tN\tsample\n"
    )

    rows = rows_from_onet_titles(src)

    assert rows[0].runtime_type == "profession"
    assert rows[0].surface == "news analysts, reporters, and journalists"
    assert "reporter" in rows[0].aliases
    assert "media worker" in rows[0].levels


def test_onet_job_titles_to_profession_rows(tmp_path):
    src = tmp_path / "Job Titles.txt"
    src.write_text(
        "O*NET-SOC Code\tJob Title\tShort Title\tSource(s)\n"
        "23-1011.00\tCriminal Lawyer\tn/a\t08\n"
    )

    rows = rows_from_onet_job_titles(src)

    assert rows[0].runtime_type == "profession"
    assert rows[0].surface == "criminal lawyer"
    assert rows[0].levels == ["legal professional", "professional worker"]
    assert rows[0].source_ids == ["onet-job-title:23-1011.00"]


def test_onet_job_titles_use_soc_major_group_for_profession_levels(tmp_path):
    src = tmp_path / "Job Titles.txt"
    src.write_text(
        "O*NET-SOC Code\tJob Title\tShort Title\tSource(s)\n"
        "15-1252.00\tSoftware Engineer\tn/a\t08\n"
    )

    rows = rows_from_onet_job_titles(src)

    assert rows[0].surface == "software engineer"
    assert rows[0].levels == ["computer and mathematical occupation", "professional worker"]


def test_onet_job_titles_skip_surfaces_spanning_multiple_soc_major_groups(tmp_path):
    src = tmp_path / "Job Titles.txt"
    src.write_text(
        "O*NET-SOC Code\tJob Title\tShort Title\tSource(s)\n"
        "15-1251.00\tEngineer\tn/a\t08\n"
        "17-2051.00\tEngineer\tn/a\t08\n"
        "15-1252.00\tSoftware Engineer\tn/a\t08\n"
    )

    rows = rows_from_onet_job_titles(src)

    assert [r.surface for r in rows] == ["software engineer"]


def test_isco_csv_to_profession_rows(tmp_path):
    src = tmp_path / "isco.csv"
    src.write_text(
        "code,title,major_group\n"
        "2211,Generalist medical practitioners,Professionals\n"
    )

    rows = rows_from_isco_csv(src)

    assert rows[0].surface == "generalist medical practitioners"
    assert rows[0].levels == ["professional worker"]


def test_esco_rdf_to_profession_rows(tmp_path):
    src = tmp_path / "esco.rdf"
    src.write_text(
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:skos="http://www.w3.org/2004/02/skos/core#" '
        'xmlns:esco="http://data.europa.eu/esco/model#">'
        '<skos:Concept rdf:about="http://data.europa.eu/esco/occupation/abc">'
        '<rdf:type rdf:resource="http://data.europa.eu/esco/model#Occupation"/>'
        '<skos:broader><skos:Concept rdf:about="http://data.europa.eu/esco/isco/C2">'
        '<skos:prefLabel xml:lang="en">Professionals</skos:prefLabel>'
        '</skos:Concept></skos:broader>'
        '<skos:prefLabel xml:lang="en">criminal lawyer</skos:prefLabel>'
        '<skos:altLabel xml:lang="en">defense attorney</skos:altLabel>'
        '</skos:Concept></rdf:RDF>'
    )

    rows = rows_from_esco_rdf(src)

    assert rows[0].runtime_type == "profession"
    assert rows[0].surface == "criminal lawyer"
    assert "defense attorney" in rows[0].aliases
    assert rows[0].levels == ["legal professional", "professional worker"]
    assert rows[0].source_ids == ["esco:abc"]


def test_esco_rdf_uses_referenced_isco_group_for_profession_levels(tmp_path):
    src = tmp_path / "esco.rdf"
    src.write_text(
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:skos="http://www.w3.org/2004/02/skos/core#">'
        '<skos:Concept rdf:about="http://data.europa.eu/esco/occupation/dev">'
        '<skos:broader rdf:resource="http://data.europa.eu/esco/isco/C2512"/>'
        '<skos:prefLabel xml:lang="en">software developer</skos:prefLabel>'
        '</skos:Concept></rdf:RDF>'
    )

    rows = rows_from_esco_rdf(src)

    assert rows[0].surface == "software developer"
    assert rows[0].levels == ["professional worker"]


def test_obo_rows_use_synonyms_and_family_roots(tmp_path):
    src = tmp_path / "doid.obo"
    src.write_text(
        "[Term]\n"
        "id: DOID:9351\n"
        "name: diabetes mellitus\n"
        "synonym: \"diabetes\" EXACT []\n"
        "is_a: DOID:28 ! endocrine system disease\n"
        "\n"
        "[Term]\n"
        "id: DOID:28\n"
        "name: endocrine system disease\n"
    )

    rows = rows_from_obo(src, "health-condition", {"DOID:28": "endocrine condition"})

    row = next(r for r in rows if r.surface == "diabetes mellitus")
    assert "diabetes" in row.aliases
    assert row.levels == ["endocrine condition"]


def test_obo_rows_use_transitive_family_roots(tmp_path):
    src = tmp_path / "doid.obo"
    src.write_text(
        "[Term]\n"
        "id: DOID:9352\n"
        "name: type 2 diabetes mellitus\n"
        "synonym: \"T2D\" EXACT []\n"
        "is_a: DOID:9351 ! diabetes mellitus\n"
        "\n"
        "[Term]\n"
        "id: DOID:9351\n"
        "name: diabetes mellitus\n"
        "is_a: DOID:28 ! endocrine system disease\n"
        "\n"
        "[Term]\n"
        "id: DOID:28\n"
        "name: endocrine system disease\n"
    )

    rows = rows_from_obo(src, "health-condition", {"DOID:28": "endocrine condition"})

    row = next(r for r in rows if r.surface == "type 2 diabetes mellitus")
    assert "t2d" in row.aliases
    assert row.levels == ["endocrine condition"]


def test_openfda_ndc_rows_use_names_only(tmp_path):
    src = tmp_path / "drug-ndc.json.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("drug-ndc-0001-of-0001.json", json.dumps({
            "results": [{
                "product_ndc": "0002-8215",
                "brand_name": "Glucophage",
                "generic_name": "METFORMIN HYDROCHLORIDE",
                "dosage_form": "TABLET",
                "route": ["ORAL"],
                "active_ingredients": [{"name": "METFORMIN HYDROCHLORIDE", "strength": "500 mg/1"}],
            }]
        }))

    rows = rows_from_openfda_ndc_zip(src)
    by_surface = {r.surface: r for r in rows}

    assert by_surface["glucophage"].runtime_type == "drug"
    assert by_surface["glucophage"].levels == ["medication"]
    assert "metformin hydrochloride" in by_surface["glucophage"].aliases
    assert "tablet" not in by_surface
    assert "oral" not in by_surface
    assert by_surface["metformin hydrochloride"].source_ids == ["openfda-ndc:0002-8215"]


def test_icd10_pcs_order_rows_to_medical_procedure_profiles(tmp_path):
    src = tmp_path / "icd10pcs_order.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr(
            "icd10pcs_order_2026.txt",
            "00001 0DBJ0ZZ 1 Excision Appendix, Open Approach\n"
            "00002 B030YZZ 1 Magnetic Resonance Imaging Brain\n",
        )

    rows = rows_from_icd10_pcs_order_zip(src)
    by_surface = {r.surface: r for r in rows}

    assert by_surface["excision appendix open approach"].runtime_type == "medical-procedure"
    assert by_surface["excision appendix open approach"].levels == ["medical and surgical procedure", "medical procedure"]
    assert by_surface["magnetic resonance imaging brain"].levels == ["imaging procedure", "medical procedure"]
    assert by_surface["excision appendix open approach"].source_ids == ["icd10pcs:0DBJ0ZZ"]


def test_icd10_pcs_order_prefers_long_title_column(tmp_path):
    src = tmp_path / "icd10pcs_order.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr(
            "icd10pcs_order_2026.txt",
            "00001 0DBJ0ZZ 1 Excision Appendix, Open Approach                        Excision of Appendix, Open Approach\n",
        )

    rows = rows_from_icd10_pcs_order_zip(src)
    surfaces = {r.surface for r in rows}

    assert "excision of appendix open approach" in surfaces
    assert "excision appendix open approach" not in surfaces


def test_nppes_rows_to_medical_facility_profiles(tmp_path):
    src = tmp_path / "nppes.zip"
    csv_text = (
        "NPI,Entity Type Code,Provider Organization Name (Legal Business Name),"
        "Provider Other Organization Name,Healthcare Provider Taxonomy Code_1\n"
        "1234567890,2,Example General Hospital,Example Hospital,282N00000X\n"
        "1234567891,1,Jane Smith,,207Q00000X\n"
    )
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("npidata_pfile_20260608-20260705.csv", csv_text)

    rows = rows_from_nppes_zip(src)
    by_surface = {r.surface: r for r in rows}

    assert by_surface["example general hospital"].runtime_type == "organization-medical-facility"
    assert by_surface["example general hospital"].aliases == ["example hospital"]
    assert by_surface["example general hospital"].levels == ["hospital", "healthcare organization"]
    assert "jane smith" not in by_surface


def test_nppes_rows_filter_noisy_org_surfaces_and_aliases(tmp_path):
    src = tmp_path / "nppes.zip"
    csv_text = (
        "NPI,Entity Type Code,Provider Organization Name (Legal Business Name),"
        "Provider Other Organization Name,Healthcare Provider Taxonomy Code_1\n"
        "1234567890,2,\"20/20 Icare, PLLC\",<UNAVAIL>,261Q00000X\n"
        "1234567891,2,Evergreen Clinic,<UNAVAIL>,261Q00000X\n"
    )
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("npidata_pfile_20260608-20260705.csv", csv_text)

    rows = rows_from_nppes_zip(src)
    by_surface = {r.surface: r for r in rows}

    assert "20/20 icare, pllc" not in by_surface
    assert by_surface["evergreen clinic"].aliases == []


def test_categorical_alias_rows_have_no_levels():
    rows = alias_rows("marital-status", {"married": ["wedded", "spouse"]})

    assert rows[0].runtime_type == "marital-status"
    assert rows[0].surface == "married"
    assert rows[0].aliases == ["wedded", "spouse"]
    assert rows[0].levels == []


def test_cldr_territories_to_nationality_rows(tmp_path):
    src = tmp_path / "territories.json"
    src.write_text(json.dumps({
        "main": {"en": {"localeDisplayNames": {"territories": {
            "150": "Europe",
            "DE": "Germany",
            "DE-alt-variant": "Federal Republic of Germany",
        }}}}
    }))
    containment = {
        "150": {"_contains": ["155"]},
        "155": {"_contains": ["DE"]},
    }

    rows = rows_from_cldr_territories(src, containment)

    row = next(r for r in rows if r.surface == "german")
    assert row.runtime_type == "nationality"
    assert "germany" in row.aliases
    assert "from germany" in row.aliases
    assert "citizen of germany" in row.aliases
    assert "federal republic of germany" in row.aliases
    assert row.levels == ["western european nationality", "european nationality"]
    assert row.source_ids == ["cldr:DE"]


def test_cldr_territories_use_demonym_as_surface(tmp_path):
    src = tmp_path / "territories.json"
    src.write_text(json.dumps({
        "main": {"en": {"localeDisplayNames": {"territories": {
            "150": "Europe",
            "DE": "Germany",
            "FR": "France",
        }}}}
    }))
    containment = {"150": {"_contains": ["DE", "FR"]}}

    rows = rows_from_cldr_territories(src, containment)
    by_surface = {r.surface: r for r in rows}

    # surface is the demonym; country name and "from/citizen of X" phrasings are aliases
    assert "germany" not in by_surface and "france" not in by_surface
    assert set(by_surface["german"].aliases) >= {"germany", "from germany", "citizen of germany"}
    assert set(by_surface["french"].aliases) >= {"france", "from france", "citizen of france"}


def test_cldr_territory_parent_prefers_geographic_region_over_later_grouping(tmp_path):
    src = tmp_path / "territories.json"
    src.write_text(json.dumps({
        "main": {"en": {"localeDisplayNames": {"territories": {
            "150": "Europe",
            "DE": "Germany",
            "EU": "European Union",
        }}}}
    }))
    containment = {
        "150": {"_contains": ["DE"]},
        "001-status-grouping": {"_contains": ["EU"]},
        "EU": {"_contains": ["DE"]},
    }

    rows = rows_from_cldr_territories(src, containment)

    assert next(r for r in rows if r.surface == "german").levels == ["european nationality"]


def test_wikidata_sparql_xml_to_demographic_rows(tmp_path):
    src = tmp_path / "seeds.xml"
    src.write_text(
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<sparql xmlns='http://www.w3.org/2005/sparql-results#'><results>"
        "<result>"
        "<binding name='type'><literal>religion</literal></binding>"
        "<binding name='item'><uri>http://www.wikidata.org/entity/Q9268</uri></binding>"
        "<binding name='itemLabel'><literal xml:lang='en'>Judaism</literal></binding>"
        "<binding name='alias'><literal xml:lang='en'>Jewish faith</literal></binding>"
        "</result>"
        "</results></sparql>"
    )

    rows = rows_from_wikidata_sparql_xml(src)

    assert rows[0].runtime_type == "religion"
    assert rows[0].surface == "judaism"
    assert rows[0].aliases == ["jewish faith"]
    assert rows[0].levels == ["abrahamic religion", "religious tradition"]
    assert rows[0].source_ids == ["wikidata:Q9268"]


def test_arda_variable_labels_to_religion_rows():
    rows = rows_from_arda_variable_labels({
        "catpp": "1200-Population of Catholics",
        "catpc": "1200-Percentage of Catholics",
        "nrepp": "9000-Population of Not Religious",
        "csynds": "1900-Description of Christian Syncretics (string)",
    })

    surfaces = {r.surface: r for r in rows}
    assert surfaces["catholics"].levels == ["christian religion", "religious tradition"]
    assert surfaces["not religious"].levels == ["nonreligion"]
    assert "arda:1200" in surfaces["catholics"].source_ids


def test_geonames_rows_include_countries_and_cities_without_self_leaking_region(tmp_path):
    geo = tmp_path / "geonames"
    geo.mkdir()
    (geo / "countryInfo.txt").write_text(
        "# header\n"
        "DE\tDEU\t276\tGM\tGermany\tBerlin\t357021\t83240000\tEU\t.de\tEUR\tEuro\t49\t#####\t^\\d{5}$\tde\t2921044\tAT,BE,CH,CZ,DK,FR,LU,NL,PL\t\n"
        "US\tUSA\t840\tUS\tUnited States\tWashington\t9629091\t327167434\tNA\t.us\tUSD\tDollar\t1\t#####-####\t^\\d{5}(-\\d{4})?$\ten-US\t6252001\tCA,MX,CU\t\n"
    )
    (geo / "admin1CodesASCII.txt").write_text("US.NY\tNew York\tNew York\t5128638\n")
    (geo / "cities500.txt").write_text(
        "5128581\tNew York City\tNew York City\tNYC,New York\t40.71427\t-74.00597\tP\tPPL\tUS\t\tNY\t\t\t\t8175133\t\t10\tAmerica/New_York\t2024-01-01\n"
    )

    rows = rows_from_geonames(geo)
    by_surface = {r.surface: r for r in rows}

    assert by_surface["germany"].levels == ["a country in europe"]
    assert "de" in by_surface["germany"].aliases
    assert by_surface["new york city"].levels == [
        "a city in new york",
        "a city in united states",
        "a city in north america",
    ]
    assert "new york" in by_surface["new york city"].aliases


def test_geonames_city_rows_keep_most_populous_duplicate_surface(tmp_path):
    geo = tmp_path / "geonames"
    geo.mkdir()
    (geo / "countryInfo.txt").write_text(
        "# header\n"
        "US\tUSA\t840\tUS\tUnited States\tWashington\t9629091\t327167434\tNA\t.us\tUSD\tDollar\t1\t#####-####\t^\\d{5}(-\\d{4})?$\ten-US\t6252001\tCA,MX,CU\t\n"
        "CA\tCAN\t124\tCA\tCanada\tOttawa\t9984670\t37058856\tNA\t.ca\tCAD\tDollar\t1\t#####\t^[ABCEGHJ-NPRSTVXY]\\d[ABCEGHJ-NPRSTV-Z][ -]?\\d[ABCEGHJ-NPRSTV-Z]\\d$\ten-CA\t6251999\tUS\t\n"
    )
    (geo / "admin1CodesASCII.txt").write_text(
        "US.IL\tIllinois\tIllinois\t4896861\n"
        "CA.08\tOntario\tOntario\t6093943\n"
    )
    (geo / "cities500.txt").write_text(
        "4250542\tSpringfield\tSpringfield\t\t39.80172\t-89.64371\tP\tPPLA2\tUS\t\tIL\t\t\t\t114394\t\t182\tAmerica/Chicago\t2024-01-01\n"
        "6154149\tSpringfield\tSpringfield\t\t45.344\t-75.724\tP\tPPL\tCA\t\t08\t\t\t\t1000\t\t93\tAmerica/Toronto\t2024-01-01\n"
    )

    rows = [r for r in rows_from_geonames(geo) if r.surface == "springfield"]

    assert len(rows) == 1
    row = rows[0]
    assert row.source_ids == ["geonames:4250542"]
    assert row.levels == [
        "a city in illinois",
        "a city in united states",
        "a city in north america",
    ]


def test_legacy_teacher_cache_org_rows_are_conservatively_imported(tmp_path):
    cache = tmp_path / "lattice_cache.json"
    cache.write_text(json.dumps({
        "goldman sachs": {"lattice": ["A financial institution", "An organization"], "tier": "e4b"},
        "psych": {"lattice": ["A field of study"], "tier": "e4b"},
        "ORG::typed bank": {"lattice": ["a financial institution"], "tier": "qwen"},
    }))

    rows = rows_from_legacy_teacher_cache(cache)
    by_surface = {r.surface: r for r in rows}

    assert by_surface["goldman sachs"].runtime_type == "ORG"
    assert by_surface["goldman sachs"].levels == ["a financial institution", "an organization"]
    assert by_surface["typed bank"].runtime_type == "ORG"
    assert "psych" not in by_surface


def test_legacy_teacher_cache_skips_weak_untyped_org_like_rows(tmp_path):
    cache = tmp_path / "lattice_cache.json"
    cache.write_text(json.dumps({
        "reddit": {"lattice": ["a platform", "a service"], "tier": "e4b"},
        "starbucks": {"lattice": ["a coffee shop", "a retail chain"], "tier": "e4b"},
        "uni": {"lattice": ["an educational institution"], "tier": "e4b"},
        "unicorn breeding start-ups": {"lattice": ["a company", "an organization"], "tier": "e4b"},
        "ORG::reddit": {"lattice": ["a platform", "an organization"], "tier": "qwen"},
    }))

    rows = rows_from_legacy_teacher_cache(cache)
    by_surface = {r.surface: r for r in rows}

    assert set(by_surface) == {"reddit"}
    assert by_surface["reddit"].source_ids == ["legacy-teacher-cache:reddit"]


def test_merge_rows_combines_aliases_levels_and_source_ids():
    artifact = merge_rows([
        ProfileRow("profession", "journalist", ["reporter"], ["media worker"], ["esco:1"], 1000.0),
        ProfileRow("profession", "journalist", ["correspondent"], ["media worker"], ["onet:2"], 1200.0),
    ])

    row = artifact["profiles"]["profession"]["journalist"]
    assert artifact["schema_version"] == 1
    assert row["aliases"] == ["correspondent", "reporter"]
    assert row["levels"] == ["media worker"]
    assert row["source_ids"] == ["esco:1", "onet:2"]
    assert row["count"] == 1200.0


def test_merge_rows_drops_profession_rows_with_only_worker_level():
    artifact = merge_rows([
        ProfileRow("profession", "software engineer", [], ["worker"], ["onet-job-title:15-1252.00"]),
        ProfileRow("profession", "journalist", [], ["media worker", "professional worker"], ["onet:27-3023.00"]),
    ])

    assert "software engineer" not in artifact["profiles"]["profession"]
    assert artifact["profiles"]["profession"]["journalist"]["levels"] == ["media worker", "professional worker"]


def test_merge_rows_drops_profession_surfaces_longer_than_two_words_but_keeps_aliases():
    artifact = merge_rows([
        ProfileRow(
            "profession",
            "software developer",
            ["senior software application developer"],
            ["computer and mathematical occupation", "professional worker"],
            ["esco:1"],
        ),
        ProfileRow(
            "profession",
            "software application developer",
            ["application developer"],
            ["computer and mathematical occupation", "professional worker"],
            ["esco:2"],
        ),
    ])

    professions = artifact["profiles"]["profession"]
    assert "software developer" in professions
    assert professions["software developer"]["aliases"] == ["senior software application developer"]
    assert "software application developer" not in professions


def test_merge_rows_drops_noisy_health_condition_surfaces():
    artifact = merge_rows([
        ProfileRow("health-condition", "asthma", [], ["respiratory condition"], ["DOID:2841"]),
        ProfileRow(
            "health-condition",
            "disease with comma, subtype",
            [],
            ["metabolic condition"],
            ["DOID:comma"],
        ),
        ProfileRow(
            "health-condition",
            "ataxia-oculomotor apraxia 3",
            [],
            ["neurological condition"],
            ["DOID:number"],
        ),
        ProfileRow(
            "health-condition",
            "very long condition name",
            [],
            ["metabolic condition"],
            ["DOID:long"],
        ),
    ])

    health = artifact["profiles"]["health-condition"]
    assert "asthma" in health
    assert "disease with comma, subtype" not in health
    assert "ataxia-oculomotor apraxia 3" not in health
    assert "very long condition name" not in health


def test_merge_rows_drops_self_leaking_levels_but_keeps_useful_parents():
    artifact = merge_rows([
        ProfileRow(
            "profession",
            "engineer",
            [],
            ["architecture and engineering occupation", "professional worker"],
            ["onet-job-title:17-2199.00"],
        ),
    ])

    assert artifact["profiles"]["profession"]["engineer"]["levels"] == ["professional worker"]


def test_coverage_report_marks_placeholder_only_types_separately():
    art = {
        "schema_version": 1,
        "created": "2026-07-07",
        "sources": {},
        "profiles": {"profession": {"journalist": {"levels": ["media worker"]}}},
    }

    report = coverage_report(art)

    assert report["profile_counts"]["profession"] == 1
    assert "gender" in report["placeholder_only_types"]
    assert "demographic-other" in report["placeholder_first_types"]


def test_build_lattice_profiles_cli_smoke(tmp_path):
    raw = tmp_path / "raw"
    (raw / "onet").mkdir(parents=True)
    (raw / "onet" / "Alternate Titles.txt").write_text(
        "O*NET-SOC Code\tTitle\tAlternate Title\tShort Title\tSource(s)\n"
        "27-3023.00\tJournalists\tReporter\tN\tsample\n"
    )
    out = tmp_path / "profiles.json"
    cov = tmp_path / "coverage.json"

    subprocess.run(
        [
            ".venv/bin/python",
            "-u",
            "scripts/build_lattice_profiles.py",
            "--raw-dir",
            str(raw),
            "--geo-dir",
            str(tmp_path / "missing-geonames"),
            "--teacher-cache",
            str(tmp_path / "missing-lattice-cache.json"),
            "--out",
            str(out),
            "--coverage-out",
            str(cov),
        ],
        check=True,
        env={"PYTHONPATH": "src:scripts"},
    )

    art = json.loads(out.read_text())
    assert art["profiles"]["profession"]
    assert json.loads(cov.read_text())["profile_counts"]["profession"] >= 1


def test_collect_rows_uses_downloaded_raw_source_layout(tmp_path):
    raw = tmp_path / "raw"
    (raw / "onet").mkdir(parents=True)
    (raw / "onet" / "Job Titles.txt").write_text(
        "O*NET-SOC Code\tJob Title\tShort Title\tSource(s)\n"
        "23-1011.00\tCriminal Lawyer\tn/a\t08\n"
    )
    (raw / "profession").mkdir()
    with zipfile.ZipFile(raw / "profession" / "esco_v1.2.0_classification_rdf.zip", "w") as zf:
        zf.writestr(
            "esco-v1.2.0.rdf",
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
            'xmlns:skos="http://www.w3.org/2004/02/skos/core#" '
            'xmlns:esco="http://data.europa.eu/esco/model#">'
            '<skos:Concept rdf:about="http://data.europa.eu/esco/occupation/abc">'
            '<rdf:type rdf:resource="http://data.europa.eu/esco/model#Occupation"/>'
            '<skos:prefLabel xml:lang="en">defense attorney</skos:prefLabel>'
            '</skos:Concept></rdf:RDF>',
        )
    (raw / "nationality").mkdir()
    with zipfile.ZipFile(raw / "nationality" / "cldr-48.2.0-json-full.zip", "w") as zf:
        zf.writestr("cldr-localenames-full/main/en/territories.json", json.dumps({
            "main": {"en": {"localeDisplayNames": {"territories": {"150": "Europe", "DE": "Germany"}}}}
        }))
        zf.writestr("cldr-core/supplemental/territoryContainment.json", json.dumps({
            "supplemental": {"territoryContainment": {"150": {"_contains": ["DE"]}}}
        }))
    (raw / "wikidata").mkdir()
    (raw / "wikidata" / "lattice_seeds.xml").write_text(
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<sparql xmlns='http://www.w3.org/2005/sparql-results#'><results>"
        "<result>"
        "<binding name='type'><literal>religion</literal></binding>"
        "<binding name='item'><uri>http://www.wikidata.org/entity/Q9268</uri></binding>"
        "<binding name='itemLabel'><literal xml:lang='en'>Judaism</literal></binding>"
        "</result>"
        "</results></sparql>"
    )
    (raw / "drug").mkdir()
    with zipfile.ZipFile(raw / "drug" / "openfda_ndc.json.zip", "w") as zf:
        zf.writestr("drug-ndc-0001-of-0001.json", json.dumps({
            "results": [{
                "product_ndc": "0002-8215",
                "brand_name": "Glucophage",
                "generic_name": "METFORMIN HYDROCHLORIDE",
                "active_ingredients": [{"name": "METFORMIN HYDROCHLORIDE"}],
            }]
        }))
    (raw / "procedure").mkdir()
    with zipfile.ZipFile(raw / "procedure" / "icd10pcs_order_2026.zip", "w") as zf:
        zf.writestr("icd10pcs_order_2026.txt", "00001 0DBJ0ZZ 1 Excision Appendix, Open Approach\n")
    (raw / "org").mkdir()
    with zipfile.ZipFile(raw / "org" / "nppes_weekly_v2.zip", "w") as zf:
        zf.writestr(
            "npidata_pfile_20260608-20260705.csv",
            "NPI,Entity Type Code,Provider Organization Name (Legal Business Name),"
            "Provider Other Organization Name,Healthcare Provider Taxonomy Code_1\n"
            "1234567890,2,Example General Hospital,Example Hospital,282N00000X\n",
        )

    rows = collect_rows(raw)
    surfaces = {r.surface for r in rows}

    assert "criminal lawyer" in surfaces
    assert "defense attorney" in surfaces
    assert "german" in surfaces
    assert "judaism" in surfaces
    assert "glucophage" in surfaces
    assert "metformin hydrochloride" in surfaces
    assert "excision appendix open approach" in surfaces
    assert "example general hospital" in surfaces


def test_collect_rows_can_include_geonames_and_legacy_teacher_cache(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    geo = tmp_path / "geonames"
    geo.mkdir()
    (geo / "countryInfo.txt").write_text(
        "# header\n"
        "DE\tDEU\t276\tGM\tGermany\tBerlin\t357021\t83240000\tEU\t.de\tEUR\tEuro\t49\t#####\t^\\d{5}$\tde\t2921044\tAT,BE,CH,CZ,DK,FR,LU,NL,PL\t\n"
    )
    cache = tmp_path / "lattice_cache.json"
    cache.write_text(json.dumps({
        "goldman sachs": {"lattice": ["A financial institution", "An organization"], "tier": "e4b"}
    }))

    rows = collect_rows(raw, geo, cache)
    by_type_surface = {(r.runtime_type, r.surface): r for r in rows}

    assert by_type_surface[("LOC", "germany")].levels == ["a country in europe"]
    assert by_type_surface[("ORG", "goldman sachs")].levels == [
        "a financial institution",
        "an organization",
    ]


def test_fetch_lattice_sources_dry_run_reports_open_and_manual_sources(tmp_path):
    report = source_report()

    assert "disease-ontology" in report["downloadable"]
    assert "hancestro" in report["downloadable"]
    assert "onet" in report["downloadable"]
    assert "esco" in report["downloadable"]
    assert "arda" in report["downloadable"]
    assert "cldr-json-full" in report["downloadable"]
    assert "openfda-ndc" in report["downloadable"]
    assert "icd10-pcs-order" in report["downloadable"]
    assert "nppes-weekly" in report["downloadable"]
    assert "wikidata-lattice-seeds" in report["downloadable"]
    assert "umls" in report["manual_or_credentialed"]
    assert "icd11" in report["manual_or_credentialed"]
    assert report["sources"]["umls"]["license"] == "UMLS individual license required"


def test_fetch_refuses_manual_or_credentialed_source(tmp_path):
    try:
        fetch("umls", tmp_path)
    except SystemExit as exc:
        assert "cannot be downloaded automatically" in str(exc)
    else:
        raise AssertionError("expected licensed source fetch to fail")


def test_fetch_dry_run_knows_requested_lattice_source_paths(tmp_path):
    assert fetch("onet", tmp_path, dry_run=True) == tmp_path / "onet" / "Job Titles.txt"
    assert fetch("esco", tmp_path, dry_run=True) == tmp_path / "profession" / "esco_v1.2.0_classification_rdf.zip"
    assert fetch("arda", tmp_path, dry_run=True) == tmp_path / "religion" / "arda_rcsdem2_stata.dta"
    assert fetch("cldr-json-full", tmp_path, dry_run=True) == tmp_path / "nationality" / "cldr-48.2.0-json-full.zip"
    assert fetch("wikidata-lattice-seeds", tmp_path, dry_run=True) == tmp_path / "wikidata" / "lattice_seeds.xml"
    assert fetch("openfda-ndc", tmp_path, dry_run=True) == tmp_path / "drug" / "openfda_ndc.json.zip"
    assert fetch("icd10-pcs-order", tmp_path, dry_run=True) == tmp_path / "procedure" / "icd10pcs_order_2026.zip"
    assert fetch("nppes-weekly", tmp_path, dry_run=True) == tmp_path / "org" / "nppes_weekly_v2.zip"


def test_populate_lattice_profiles_writes_artifact_and_missing_source_report(tmp_path):
    raw = tmp_path / "raw"
    (raw / "onet").mkdir(parents=True)
    (raw / "onet" / "Alternate Titles.txt").write_text(
        "O*NET-SOC Code\tTitle\tAlternate Title\tShort Title\tSource(s)\n"
        "27-3023.00\tJournalists\tReporter\tN\tsample\n"
    )
    out = tmp_path / "profiles.json"
    cov = tmp_path / "coverage.json"
    report = tmp_path / "population.json"

    subprocess.run(
        [
            ".venv/bin/python",
            "-u",
            "scripts/populate_lattice_profiles.py",
            "--raw-dir",
            str(raw),
            "--geo-dir",
            str(tmp_path / "missing-geonames"),
            "--teacher-cache",
            str(tmp_path / "missing-lattice-cache.json"),
            "--out",
            str(out),
            "--coverage-out",
            str(cov),
            "--report-out",
            str(report),
        ],
        check=True,
        env={"PYTHONPATH": "src:scripts"},
    )

    art = json.loads(out.read_text())
    pop = json.loads(report.read_text())
    assert art["profiles"]["profession"]
    assert json.loads(cov.read_text())["profile_counts"]["profession"] >= 1
    assert "onet-alternate-titles" in pop["available_sources"]
    assert "isco08" in pop["missing_sources"]
    assert "esco-rdf" in pop["missing_sources"]
    assert "esco" not in pop["unimplemented_sources"]
    assert pop["exhaustive"] is False


def test_populate_lattice_profiles_require_exhaustive_fails_when_sources_are_missing(tmp_path):
    out = tmp_path / "profiles.json"
    report = tmp_path / "population.json"

    proc = subprocess.run(
        [
            ".venv/bin/python",
            "-u",
            "scripts/populate_lattice_profiles.py",
            "--raw-dir",
            str(tmp_path / "raw"),
            "--geo-dir",
            str(tmp_path / "missing-geonames"),
            "--teacher-cache",
            str(tmp_path / "missing-lattice-cache.json"),
            "--out",
            str(out),
            "--report-out",
            str(report),
            "--require-exhaustive",
        ],
        env={"PYTHONPATH": "src:scripts"},
        stderr=subprocess.PIPE,
        text=True,
    )

    assert proc.returncode != 0
    assert "not exhaustive" in proc.stderr
    assert json.loads(report.read_text())["exhaustive"] is False
