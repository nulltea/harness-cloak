# Lattice source cache

This directory holds raw files used by `scripts/build_lattice_profiles.py`.

Runtime substitution never reads this directory and never downloads data. Raw sources are fetched or placed
manually, then normalized into `data/lattice_profiles/fine_lattice_profiles.json`.

The normalized artifact uses the standard generalization lattice cache schema in
`docs/specs/generalization-lattice-cache.md`.

Expected paths:

- `raw/onet/Job Titles.txt`
- `raw/onet/Alternate Titles.txt`
- `raw/isco/isco.csv`
- `raw/profession/esco_v1.2.0_classification_rdf.zip`
- `raw/religion/arda_rcsdem2_stata.dta`
- `raw/nationality/cldr-48.2.0-json-full.zip`
- `raw/wikidata/lattice_seeds.xml`
- `raw/health/mondo.obo`
- `raw/health/doid.obo`

Build:

```bash
PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_lattice_profiles.py \
  --raw-dir data/lattice_sources/raw \
  --out data/lattice_profiles/fine_lattice_profiles.json
```

Population with an honesty report:

```bash
PYTHONPATH=src:scripts .venv/bin/python -u scripts/populate_lattice_profiles.py \
  --raw-dir data/lattice_sources/raw \
  --out data/lattice_profiles/fine_lattice_profiles.json \
  --coverage-out results/lattice_profile_coverage.json \
  --report-out results/lattice_profile_population.json
```

Use `--require-exhaustive` only when CI or a release gate should fail if any target source file is missing
or parser support is missing for a required downloaded-source family: O*NET job titles, ESCO RDF, ARDA,
CLDR territories, Wikidata lattice seeds, and manual categorical aliases. Optional older/manual expansion
families are still reported as `optional_missing_sources` or `unimplemented_sources`, but they do not block
the downloaded-source exhaustive build. Coverage counts are candidate source coverage, not privacy
guarantees.

Optional explicit fetch for open OBO sources:

```bash
PYTHONPATH=src:scripts .venv/bin/python -u scripts/download/fetch_lattice_sources.py \
  --source disease-ontology \
  --source mondo
```

Explicit fetch for larger public lattice sources:

```bash
PYTHONPATH=src:scripts .venv/bin/python -u scripts/download/fetch_lattice_sources.py \
  --source onet \
  --source arda \
  --source esco \
  --source cldr-json-full

PYTHONPATH=src:scripts .venv/bin/python -u scripts/download/fetch_lattice_sources.py \
  --source wikidata-lattice-seeds
```

Inspect available source actions:

```bash
PYTHONPATH=src:scripts .venv/bin/python -u scripts/download/fetch_lattice_sources.py --list
PYTHONPATH=src:scripts .venv/bin/python -u scripts/download/fetch_lattice_sources.py --all-downloadable --dry-run
```

The fetcher downloads stable public file URLs plus the explicit `wikidata-lattice-seeds` SPARQL export.
It refuses credentialed APIs and restricted datasets and prints the manual action instead.

Licensed or credentialed sources:

- O*NET: `raw/onet/Job Titles.txt` is fetched automatically. If parser coverage needs alternate titles,
  download the current database text ZIP from the O*NET Resource Center after reviewing the database
  license, then extract `Alternate Titles.txt` to `raw/onet/Alternate Titles.txt`.
- ESCO: `raw/profession/esco_v1.2.0_classification_rdf.zip` is fetched automatically from the public RDF
  package. If parser coverage needs the generated English CSV package, use the ESCO download page and put
  the export under `raw/profession/`.
- ICD-11: register for WHO ICD API credentials, export approved local JSONL/CSV, and never commit client
  secrets.
- UMLS: use an individual UTS/UMLS license and keep restricted exports out of git.
- Wikidata/SPARQL: `raw/wikidata/lattice_seeds.xml` is fetched automatically for a small approved seed query.
  Larger exports should come from dumps or an approved offline SPARQL job.
- ARDA: `raw/religion/arda_rcsdem2_stata.dta` is fetched automatically from the public OSF download URL.
