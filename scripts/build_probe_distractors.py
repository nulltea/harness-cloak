"""Build the walk_risk distractor pools from the arms artifact.

Pools are the same-type anonymity sets the contrastive re-identification probe softmaxes over
(cloak/probe.py). Corpus-empirical: every detected surface of a type across the task corpora is
a plausible distractor for that type. Rebuild after corpus changes; consumers reload lazily.

Run: PYTHONPATH=src:scripts .venv/bin/python scripts/build_probe_distractors.py
"""
import json
from pathlib import Path

from build_arms_artifact import load_artifact

OUT = Path("data/probe_distractors.json")


def main():
    pools: dict[str, set] = {}
    art = load_artifact()
    for corpus in art.values():
        for arms in corpus.values():
            for e in arms["tau_walk"][1]:
                pools.setdefault(e["type"], set()).add(e["surface"])
    out = {t: sorted(s) for t, s in pools.items()}
    OUT.write_text(json.dumps(out, indent=1))
    print({t: len(v) for t, v in out.items()}, "->", OUT)


if __name__ == "__main__":
    main()
