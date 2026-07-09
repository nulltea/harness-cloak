"""Post-fix end-to-end verification of the detector config-surface fixes (branch
detector-config-surface-fixes). Companion to detector_config_surface_probes.py, which froze the
PRE-fix behavior; this drives the real Detector(profile="clinical") on the same adversarial
inputs and asserts the fixed behavior.

    PYTHONPATH=src .venv/bin/python -u scripts/spikes/detector_fixes_verify.py
"""
from __future__ import annotations

from cloak.detect import Detector, Span, _dedupe, coref_chains

det = Detector(profile="clinical")
print(f"loaded profile={det.profile.name} stop_words={len(det.stop_words)} max_words={det.max_words}")

failures = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        failures.append(name)


# 1. clinical profile: "RN" span survives the stop-word filter (was silently deleted)
spans = det.detect("The RN administered the evening medication to the patient.")
rn = [s for s in spans if s.text.lower() == "rn"]
check("rn-not-deleted", not ("rn" in det.stop_words), f"'rn' in stop_words={'rn' in det.stop_words}; RN spans detected={len(rn)}")

# 2. clinical profile: blood pressure 120/80 is not a CODE ref (custom REF_CODE not registered)
spans = det.detect("BP was 120/80 today and stable.")
bp_code = [s for s in spans if "120/80" in s.text and s.type == "CODE"]
check("bp-not-code", not bp_code, f"CODE spans on '120/80': {[(s.text, s.source) for s in bp_code]}")

# 3. dedupe: precise pattern span survives a wide low-confidence span (unit-verified; real-flow probe)
wide = Span(0, 30, "wide low-confidence misc span!", "MISC", 0.31, "gliner")
ssn = Span(10, 21, "123-45-6789", "CODE", 0.99, "presidio-pattern")
kept = _dedupe([wide, ssn])
check("dedupe-keeps-precise", [s.type for s in kept] == ["CODE"], f"kept={[(s.type, s.score) for s in kept]}")

# 4. coref: family members stay distinct identities
text = "Anna Smith met Peter Smith. Later Smith left."
spans = [Span(0, 10, "Anna Smith", "PERSON", 0.9, "gliner"),
         Span(15, 26, "Peter Smith", "PERSON", 0.9, "gliner"),
         Span(34, 39, "Smith", "PERSON", 0.9, "gliner")]
coref_chains(text, spans)
check("coref-no-identity-merge", spans[0].chain != spans[1].chain and spans[2].chain == spans[1].chain,
      f"chains={[(s.text, s.chain) for s in spans]}")

# 5. real-model detection still works end to end (control)
spans = det.detect("Sarah Johnson, a cardiologist in Oslo, was seen on March 3, 2021.")
types = {s.type for s in spans}
check("controls-detected", {"PERSON", "LOC", "DATETIME"} <= types, f"types={sorted(types)}")

print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
raise SystemExit(1 if failures else 0)
