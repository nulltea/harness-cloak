# Lattice source cache

This directory holds raw files used by `scripts/build_lattice_profiles.py`.

Runtime substitution never reads this directory and never downloads data. Raw sources are fetched or placed
manually, then normalized into `data/lattice_profiles/fine_lattice_profiles.json`.

Expected paths:

- `raw/onet/Alternate Titles.txt`
- `raw/isco/isco.csv`
- `raw/health/mondo.obo`
- `raw/health/doid.obo`

Build:

```bash
PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_lattice_profiles.py \
  --raw-dir data/lattice_sources/raw \
  --out data/lattice_profiles/fine_lattice_profiles.json
```

Optional explicit fetch for open OBO sources:

```bash
PYTHONPATH=src:scripts .venv/bin/python -u scripts/fetch_lattice_sources.py \
  --source disease-ontology \
  --source mondo
```

O*NET, ESCO, ISCO, UMLS, ICD-11, Wikidata, and ARDA may have terms, licenses, credentials, or rate limits.
Place their exported files manually under `raw/` unless the user explicitly approves an automated fetch path.
