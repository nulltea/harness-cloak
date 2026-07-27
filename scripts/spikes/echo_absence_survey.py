#!/usr/bin/env python3
"""Survey generalization echo absence across the corpus (pre-RL audit R2).

One BC-teacher rewrite per document through the pinned remote model, then per
generalized mention: did out_p echo the fill, did inversion restore the source,
and if absent — is an abbreviation/partial paraphrase detectable?

Run: CLOAK_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
       scripts/spikes/echo_absence_survey.py
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from cloak.concurrent import pmap
from cloak.ranker.environment import assemble_action_vector, load_ranker_environment
from cloak.ranker.interactive import behavior_clone_trajectory
from cloak.ranker.environment import LambdaProfile
from cloak.ranker.privacy import DirectCountPrivacyProvider
from cloak.reward.extract import invert
from cloak.reward.roundtrip import _remote, _template

sys.path.insert(0, "scripts")
from train_interactive_ranker import _demote_out_of_scope_decisions

OUT = Path("results/ranker_v2/architecture/echo-absence-survey.json")


def _acronym(fill: str) -> str:
    words = re.findall(r"[A-Za-z]+", fill)
    return "".join(w[0] for w in words).upper() if len(words) >= 2 else ""


def classify(fill: str, surface: str, out_p: str, out_final: str) -> str:
    low_p = out_p.lower()
    if fill.lower() in low_p:
        return "echoed"
    acronym = _acronym(fill)
    if acronym and re.search(rf"\b{re.escape(acronym)}\b", out_p):
        return "absent_abbreviated"
    words = [w for w in re.findall(r"[a-z]{4,}", fill.lower())
             if w not in ("with", "that", "this", "from")]
    if words and sum(w in low_p for w in words) >= max(1, len(words) - 1):
        return "absent_partial_words"
    return "absent_dropped"


def main() -> None:
    targets = json.loads(
        Path("results/ranker_v2/reward/profile-count-targets.json").read_text()
    )
    documents = tuple(load_ranker_environment(
        Path("results/ranker_v2/environment/ranker-env.json")
    ).values())
    documents, _ = _demote_out_of_scope_decisions(
        documents, DirectCountPrivacyProvider(targets)
    )
    lambda_zero = LambdaProfile("lambda-zero", 0.0)
    jobs = []
    for document in documents:
        trajectory = behavior_clone_trajectory(document, lambda_zero)
        doc_p, replacements = assemble_action_vector(
            document, trajectory.action_vector
        )
        jobs.append((document, doc_p, replacements))

    remote = _remote()
    outputs = pmap(
        lambda job: remote.generate(
            _template({"corpus": job[0].corpus}).format(doc=job[1])
        ),
        jobs, workers=6,
    )

    rows = []
    outcome_by_type: dict[str, Counter] = defaultdict(Counter)
    fill_length_bins: dict[str, Counter] = defaultdict(Counter)
    for (document, doc_p, replacements), out_p in zip(jobs, outputs, strict=True):
        out_final, stats = invert(out_p, replacements)
        for entry in replacements:
            if entry["action"] != "generalize":
                continue
            outcome = classify(
                entry["fill"] if "fill" in entry else entry["replacement"],
                entry["surface"], out_p, out_final,
            )
            restored = entry["surface"].lower() in out_final.lower()
            outcome_by_type[entry["type"]][outcome] += 1
            words = len(entry["replacement"].split())
            bin_name = "1w" if words == 1 else "2-3w" if words <= 3 else "4w+"
            fill_length_bins[bin_name][outcome] += 1
            rows.append({
                "doc_id": document.doc_id,
                "type": entry["type"],
                "fill_words": words,
                "outcome": outcome,
                "restored": restored,
            })

    total = Counter(row["outcome"] for row in rows)
    report = {
        "documents": len(jobs),
        "generalized_mentions": len(rows),
        "outcomes": dict(total),
        "by_runtime_type": {k: dict(v) for k, v in outcome_by_type.items()},
        "by_fill_length": {k: dict(v) for k, v in fill_length_bins.items()},
        "restored_rate_overall": sum(r["restored"] for r in rows) / max(len(rows), 1),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, sort_keys=True))
    share = {k: f"{v / len(rows):.1%}" for k, v in sorted(total.items())}
    print(f"docs {len(jobs)} | generalized mentions {len(rows)} | outcomes {share}")
    print(f"restored rate {report['restored_rate_overall']:.1%} -> {OUT}")


if __name__ == "__main__":
    main()
