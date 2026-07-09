"""Probe detector config/processing surfaces for span/classification issues (no GPU needed).

Verifies the model-independent parts of src/cloak/detect.py: chunk-boundary span loss, _dedupe
widest-wins overlap resolution, the pronoun filter vs clinical abbreviations, coref_chains
identity collapse, and the custom Presidio patterns on clinical-shaped text.

    PYTHONPATH=src .venv/bin/python scripts/spikes/detector_config_surface_probes.py
"""
from __future__ import annotations

import re

from cloak.detect import Span, _PRONOUNS, _chunks, _dedupe, coref_chains

print("== 1. chunk boundaries ==")
# entity straddling the 1200-char cut with no newline/'. ' in the second half -> hard mid-word cut
filler = "x" * 1195
text = filler + " Sarah Johnson was seen"
chunks = list(_chunks(text))
b = chunks[0][1][-10:], chunks[1][1][:15] if len(chunks) > 1 else ""
print(f"chunks={len(chunks)} cut=...{b[0]!r} | {b[1]!r}   (name straddles cut: {'Sarah' in chunks[0][1] and 'Johnson' not in chunks[0][1]})")
# '. ' cut inside an honorific
text2 = "y" * 900 + " Seen by Dr. Smith at the clinic. " + "z" * 400
chunks2 = list(_chunks(text2))
print(f"honorific cut: chunk0 ends {chunks2[0][1][-12:]!r}  ('Dr.' split from 'Smith': {chunks2[0][1].rstrip().endswith('Dr.')})")

print("\n== 2. _dedupe widest-wins ==")
wide = Span(0, 30, "wide low-confidence misc span!", "MISC", 0.31, "gliner")
precise = Span(10, 21, "123-45-6789", "CODE", 0.99, "presidio")
kept = _dedupe([wide, precise])
print(f"kept: {[(s.type, s.score) for s in kept]}   (0.99 CODE swallowed by 0.31 MISC: {len(kept) == 1 and kept[0].type == 'MISC'})")

print("\n== 3. pronoun filter vs clinical text ==")
print(f"'rn' in _PRONOUNS: {'rn' in _PRONOUNS}  -> a detected 'RN' (registered nurse) span is dropped: "
      f"{not ('RN'.lower() not in _PRONOUNS)}")

print("\n== 4. coref_chains identity collapse ==")
spans = [Span(0, 10, "Anna Smith", "PERSON", 0.9, "gliner"),
         Span(20, 31, "Peter Smith", "PERSON", 0.9, "gliner")]
coref_chains("Anna Smith met Peter Smith", spans)
print(f"chains: {[(s.text, s.chain) for s in spans]}   (two people share a chain: {spans[0].chain == spans[1].chain})")

print("\n== 5. custom Presidio patterns on clinical text ==")
ref = re.compile(r"\b\d{3,6}/\d{2,4}\b")
money_k = re.compile(r"\b\d{1,4}(?:\.\d+)?[kKmM]\b")
for s in ("BP was 120/80 today", "vision 20/40 both eyes", "budget year 2021/22", "ran a 5k on saturday", "10M view count"):
    print(f"  {s!r:32} REF_CODE={bool(ref.search(s))} MONEY={bool(money_k.search(s))}")

print("\n== 6. encoder window vs chunk size ==")
try:
    import json
    from pathlib import Path
    cfg = json.loads(Path("data/models/pii_gliner_multidomain/checkpoint-2479/gliner_config.json").read_text())
    print(f"gliner max_len={cfg.get('max_len')} tokens; chunk=1200 chars"
          f"  (spaced OCR text ~1 token/2 chars -> can exceed window; tail silently unscanned)")
except Exception as e:
    print(f"config read failed: {e}")
