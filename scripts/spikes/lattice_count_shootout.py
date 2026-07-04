"""Promotion shootout for the structural risk measure: score = 1/aset_count, evaluated
against the SAME cached attacker labels as the probe shootout (zero remote calls, zero
GPU). Decision gate for docs/research/inference-risk-enforcement.md: adopt structural
counts if level-ordering is within 0.05 of walk_risk's; otherwise counts still mask but
floors are LM-calibrated per (type, depth) offline.

Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/lattice_count_shootout.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
# ponytail: privacy_probe_shootout's top-level import references guess_back_risk, renamed to
# walk_risk in the working tree; alias it so the module (and its pure auc/rank helpers) loads.
import cloak.probe as _probe  # noqa: E402
if not hasattr(_probe, "guess_back_risk"):
    _probe.guess_back_risk = _probe.walk_risk
from privacy_probe_shootout import auc, per_span_rank_agreement  # noqa: E402

from cloak.anonymity import aset_count  # noqa: E402

report = json.loads(Path("results/privacy_probe_shootout.json").read_text())
items = report["items"]
for it in items:
    it["aset"] = aset_count(it["fill"], it["type"], it["surface"], strict=True)
    it["inv_aset"] = 1.0 / it["aset"]
    it["fail_closed"] = (it["aset"] == 1.0
                         and it["fill"].lower().strip() != it["surface"].lower().strip())

out = {"n_items": len(items), "scores": {}}
for probe in ("p4", "p6", "inv_aset"):
    out["scores"][probe] = {
        "auc_hit1": auc([it[probe] for it in items], [it["hit1"] for it in items]),
        "auc_hit5": auc([it[probe] for it in items], [it["hit5"] for it in items]),
        "level_ordering": per_span_rank_agreement(items, probe),
    }
    print(probe, out["scores"][probe], flush=True)

# calibration table: attacker hit rate per (type, count bucket) -> floor per type
buckets = defaultdict(list)
for it in items:
    b = 0 if it["aset"] < 10 else 1 if it["aset"] < 100 else 2 if it["aset"] < 1e4 else 3
    buckets[(it["type"], b)].append(it["hit5"])
table = {f"{t}|{['<10', '10-100', '100-10k', '>=10k'][b]}":
         {"n": len(v), "attacker_hit5": round(sum(v) / len(v), 3)}
         for (t, b), v in sorted(buckets.items())}
out["calibration_table"] = table
for k, v in table.items():
    print(f"{k:24s} {v}", flush=True)

# fail-closed rate per type (strict certifying mode): sizes the utility cost of parse
# misses and therefore the case for the offline LM count-estimator (build-time tool)
fc = defaultdict(lambda: [0, 0])
for it in items:
    fc[it["type"]][1] += 1
    fc[it["type"]][0] += it["fail_closed"]
out["fail_closed_rate"] = {t: {"rate": round(a / b, 3), "n": b}
                           for t, (a, b) in sorted(fc.items())}
print("fail_closed_rate:", out["fail_closed_rate"], flush=True)

Path("results/lattice_count_shootout.json").write_text(json.dumps(out, indent=1))
print("-> results/lattice_count_shootout.json")
