"""Phase 0 — build the PII span-detection training set in gliner format.

Default (no --mix): TAB-only (single-domain fine-tune v1 input).
With --mix: multi-domain mix (fine-tune v2) — TAB (anchor) + auxiliary sources mapped onto the 8 TAB
types, plus an optional diverse-label slice (Pile-NER, own labels) for open-label generality. TAB stays
dominant by a token/window cap. See research-wiki/training/2026-07-04-ft-detector-quasi.md.

Every source yields (text, [(char_start, char_end, label_phrase)]); windows are 150-word passages that
never cut inside a span (char-based chunkers split abbreviated names), words via gliner's WordsSplitter
so tokenization matches inference, subword length preflighted against the model budget.

Reproducibility: SEED fixes sampling/shuffle; HF streaming `.take(n)` order is fixed for a given
dataset revision, so a build is reproducible against a pinned revision. For a frozen final dataset,
materialize the streamed sources to local files first (the loaders read the same (text, spans) shape).

Build v1:  PYTHONPATH=src .venv/bin/python -u scripts/build_pii_span_dataset.py
Build v2:  ... --mix nemotron=8000,pilener=4000,wikibio=corpora/wikipedia_bio/train.json --out-dir data/pii_span_dataset_multidomain
Check:     ... --selfcheck
"""
import argparse
import ast
import json
import os
import random
import re
from collections import Counter

from cloak.detect import GLINER_LABELS

TYPE2PHRASE = {t: p for p, t in GLINER_LABELS.items()}   # TAB entity_type -> label phrase
WINDOW_WORDS = 150
MAX_SUBWORD = 480
MAX_SPAN_WORDS = 60
SEED = 42

# v2 mix target shares (by windows), TAB-anchored — see the training-experiment spec.
MIX_RATIO = {"tab": 0.50, "nemotron": 0.25, "wikibio": 0.15, "pilener": 0.10}

# Nemotron-PII label -> TAB entity_type. Unmapped labels are DROPPED (drop-not-invent). No MISC
# (identifying events) and no QUANTITY come from Nemotron — those stay TAB(+bio) only.
NEMOTRON_MAP = {
    "first_name": "PERSON", "last_name": "PERSON", "middle_name": "PERSON", "full_name": "PERSON",
    "name": "PERSON",
    "organization": "ORG", "company": "ORG", "employer": "ORG",
    "street": "LOC", "city": "LOC", "county": "LOC", "state": "LOC", "country": "LOC",
    "postcode": "LOC", "coordinate": "LOC", "address": "LOC", "building_number": "LOC",
    "date_time": "DATETIME", "date_of_birth": "DATETIME", "date": "DATETIME", "time": "DATETIME",
    "employee_id": "CODE", "account_number": "CODE", "certificate_license_number": "CODE",
    "ssn": "CODE", "tax_id": "CODE", "medical_record_number": "CODE",
    "health_plan_beneficiary_number": "CODE", "bank_routing_number": "CODE", "swift_bic": "CODE",
    "credit_debit_card": "CODE", "cvv": "CODE", "pin": "CODE", "password": "CODE", "api_key": "CODE",
    "user_name": "CODE", "email": "CODE", "email_address": "CODE", "phone_number": "CODE",
    "phone": "CODE", "fax_number": "CODE", "ipv4": "CODE", "ipv6": "CODE", "mac_address": "CODE",
    "device_identifier": "CODE", "vehicle_identifier": "CODE", "license_plate": "CODE",
    "biometric_identifier": "CODE", "http_cookie": "CODE", "unique_id": "CODE",
    "passport_number": "CODE", "driver_license": "CODE",
    "age": "DEM", "gender": "DEM", "race_ethnicity": "DEM", "religious_belief": "DEM",
    "political_view": "DEM", "sexuality": "DEM", "language": "DEM", "education_level": "DEM",
    "employment_status": "DEM", "blood_type": "DEM", "nationality": "DEM", "occupation": "DEM",
    "job_title": "DEM", "marital_status": "DEM",
}

_SPLIT = None


def splitter():
    global _SPLIT
    if _SPLIT is None:
        from gliner.data_processing import WordsSplitter
        _SPLIT = WordsSplitter()
    return _SPLIT


# ---------- sources: each yields (text, [(char_start, char_end, label_phrase)]) ----------

def tab_gold(doc):
    """Union of annotators' DIRECT/QUASI mentions, deduped by (start,end); DIRECT wins type ties;
    sorted for determinism."""
    best = {}
    for ann_name in sorted(doc["annotations"]):
        for m in doc["annotations"][ann_name]["entity_mentions"]:
            if m["identifier_type"] not in ("DIRECT", "QUASI"):
                continue
            key = (m["start_offset"], m["end_offset"])
            if key not in best or (m["identifier_type"] == "DIRECT"
                                   and best[key]["identifier_type"] != "DIRECT"):
                best[key] = m
    return sorted((m["start_offset"], m["end_offset"], m["entity_type"])
                  for m in best.values() if m["entity_type"] in TYPE2PHRASE)


def tab_source(path):
    for doc in json.load(open(path)):
        yield doc["text"], [(s, e, TYPE2PHRASE[t]) for s, e, t in tab_gold(doc)]


def wikibio_source(path):
    """Wikipedia-bio (Papadopoulou et al., arXiv 2205.06895) — same TAB annotation schema. Reads the
    NR-format JSON at `path` if present; skipped (with a warning) otherwise, since it is not on HF."""
    fp = path if os.path.isfile(path) else os.path.join(path, "wikipedia_bio.json")
    if not os.path.exists(fp):
        print(f"  [wikibio] SKIP — dataset not found at {path} (fetch the NR release first)")
        return
    for doc in json.load(open(fp)):
        yield doc["text"], [(s, e, TYPE2PHRASE[t]) for s, e, t in tab_gold(doc)]


def _wordchar(c):   # word boundary = alnum or underscore (isalnum misses '_')
    return c.isalnum() or c == "_"


def _spans_field(row):
    sp = row["spans"]
    return ast.literal_eval(sp) if isinstance(sp, str) else sp


def nemotron_source(n):
    """nvidia/Nemotron-PII — char-offset spans; map labels onto TAB-8, drop unmapped. Every offset is
    validated against the span's own `text` field (drop-not-trust); mismatches are counted, not used.
    Streaming .take(n) is deterministic for a fixed dataset revision (see module note)."""
    from datasets import load_dataset
    ds = load_dataset("nvidia/Nemotron-PII", split="train", streaming=True).take(n)
    bad = 0
    for row in ds:
        text, out = row["text"], []
        for s in _spans_field(row):
            if s["label"] not in NEMOTRON_MAP:
                continue
            cs, ce = s["start"], s["end"]
            if not (0 <= cs < ce <= len(text)) or (s.get("text") and text[cs:ce] != s["text"]):
                bad += 1                                    # stale/byte/misaligned offset -> drop
                continue
            out.append((cs, ce, TYPE2PHRASE[NEMOTRON_MAP[s["label"]]]))
        yield text, out
    if bad:
        print(f"  [nemotron] dropped {bad} spans failing offset validation")


def pilener_source(n):
    """Universal-NER/Pile-NER-type — generality slice. Conversation format: entity mentions per type
    as strings; spans derived by first-occurrence string match. Labels kept diverse (NOT remapped) so
    the open-label encoder keeps practicing arbitrary phrases."""
    from datasets import load_dataset
    ds = load_dataset("Universal-NER/Pile-NER-type", split="train", streaming=True).take(n)
    ambig = 0  # mentions with >1 boundary-aligned occurrence (label applied to all — noise risk)
    for row in ds:
        turns = row["conversations"]
        if not turns or "Text:" not in turns[0]["value"]:
            continue
        text = turns[0]["value"].split("Text:", 1)[1].strip()
        spans = []
        for i in range(1, len(turns) - 1):
            m = re.search(r"[Ww]hat describes (.+?) in the text\??", turns[i]["value"])
            if not m or turns[i + 1]["from"] != "gpt":
                continue
            label = m.group(1).strip().lower()
            try:
                mentions = ast.literal_eval(turns[i + 1]["value"])
            except (ValueError, SyntaxError):
                continue
            for men in mentions if isinstance(mentions, list) else []:
                if not isinstance(men, str) or not men.strip():
                    continue
                # label EVERY word-boundary-aligned exact occurrence (reject substrings inside words);
                # first-occurrence-only guessing was the flagged unsafe path
                hits = 0
                for mt in re.finditer(re.escape(men), text):
                    a, b = mt.start(), mt.end()
                    if (a > 0 and _wordchar(text[a - 1])) or (b < len(text) and _wordchar(text[b])):
                        continue
                    spans.append((a, b, label))
                    hits += 1
                ambig += hits > 1
        yield text, spans
    if ambig:
        print(f"  [pilener] {ambig} mentions had >1 occurrence (label applied to all — audit if noisy)")


# ---------- windowing (source-agnostic; spans = [(char_start, char_end, phrase)]) ----------

def _word_spans(words, spans):
    """char-offset spans -> [(word_start, word_end_inclusive, phrase)]; dedupe (word span, drop the
    2nd label on a collision), drop passage-level spans > MAX_SPAN_WORDS. Returns (kept, conflicts, dropped)."""
    seen, out, conflicts, dropped, bad = {}, [], 0, 0, []
    for cs, ce, ph in spans:
        wi = [i for i, (_, s, e) in enumerate(words) if s < ce and e > cs]
        if not wi:
            continue
        if wi[-1] - wi[0] + 1 > MAX_SPAN_WORDS:   # unrepresentable passage-level span
            dropped += 1
            bad.append((wi[0], wi[-1]))
            continue
        key = (wi[0], wi[-1])
        if key in seen:
            conflicts += (seen[key] != ph)
            continue
        seen[key] = ph
        out.append((wi[0], wi[-1], ph))
    return out, conflicts, dropped, bad


def doc_to_records(text, spans):
    """Word-window; windows never cut inside a span; bounded to ~WINDOW_WORDS by retracting the cut.
    Windows overlapping a dropped-long span are skipped (so its tokens don't become false negatives)."""
    words = list(splitter()(text))
    wspans, _, _, bad = _word_spans(words, spans)
    n, i = len(words), 0
    while i < n:
        j = min(i + WINDOW_WORDS, n)
        while j > i + 1 and any(s < j <= e for s, e, _ in wspans):
            j -= 1
        if j <= i + 1:
            over = [e for s, e, _ in wspans if s <= i and e >= i + 1]
            j = max(over) + 1 if over else i + 1
        if not any(bs < j and be >= i for bs, be in bad):   # skip windows holding an unrepresentable span
            ner = [[s - i, e - i, ph] for (s, e, ph) in wspans if i <= s and e < j]
            yield {"tokenized_text": [t for t, _, _ in words[i:j]], "ner": ner}
        i = j


def source_windows(pairs, cap=0):
    """(text, spans) iterable -> list of window records; conflict/drop counts. Optional cap by windows."""
    recs, conflicts, dropped = [], 0, 0
    for text, spans in pairs:
        _, c, d, _ = _word_spans(list(splitter()(text)), spans)
        conflicts += c
        dropped += d
        for rec in doc_to_records(text, spans):
            recs.append(rec)
            if cap and len(recs) >= cap:
                return recs, conflicts, dropped
    return recs, conflicts, dropped


# ---------- build ----------

def _tokenizer(name="microsoft/deberta-v3-base"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def _stats(tag, recs, conflicts, dropped, tok):
    types = Counter(p.split(",")[0][:14] for r in recs for _, _, p in r["ner"])
    sub = sorted(len(tok(" ".join(w["tokenized_text"]), add_special_tokens=False)["input_ids"]) for w in recs)
    over = sum(1 for s in sub if s > MAX_SUBWORD)
    p99 = sub[int(0.99 * (len(sub) - 1))] if sub else 0
    print(f"  [{tag}] {len(recs)} windows, {sum(len(r['ner']) for r in recs)} spans | "
          f"subword p99 {p99}, max {max(sub or [0])}, over {MAX_SUBWORD}: {over} | "
          f"conflicts {conflicts}, dropped_long {dropped}")
    print(f"       labels: {dict(types.most_common(8))}")
    assert over == 0, f"[{tag}] {over} windows exceed {MAX_SUBWORD} subwords"


def build(args):
    rng = random.Random(SEED)
    tok = _tokenizer()

    print(f"== source: tab ({args.train}) ==")
    tab, c, d = source_windows(tab_source(args.train))
    _stats("tab", tab, c, d, tok)
    train_recs = list(tab)

    if args.mix:
        n_tab = len(tab)
        for name, spec in args.mix.items():
            if name == "nemotron":
                pairs = nemotron_source(spec)
            elif name == "pilener":
                pairs = pilener_source(spec)
            elif name == "wikibio":
                pairs = wikibio_source(spec)
            else:
                raise SystemExit(f"unknown mix source: {name}")
            cap = int(round(MIX_RATIO[name] / MIX_RATIO["tab"] * n_tab))
            print(f"== source: {name} (target ~{cap} windows) ==")
            recs, c, d = source_windows(pairs, cap=cap * 3)   # gather headroom, then trim to cap
            recs = [r for r in recs if r["ner"]]              # aux must contribute labels, not empty negatives
            if len(recs) > cap:
                recs = rng.sample(recs, cap)
            elif name == "wikibio" and 0 < len(recs) < cap:   # oversample with replacement, ≤×2 unique
                recs = [rng.choice(recs) for _ in range(min(cap, 2 * len(recs)))]
            if recs:
                _stats(name, recs, c, d, tok)
                train_recs += recs

    rng.shuffle(train_recs)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "train.jsonl"), "w") as f:
        for r in train_recs:
            f.write(json.dumps(r) + "\n")
    # dev is always TAB-only (the selection gate scores against TAB)
    dev, c, d = source_windows(tab_source(args.dev))
    with open(os.path.join(args.out_dir, "dev.jsonl"), "w") as f:
        for r in dev:
            f.write(json.dumps(r) + "\n")
    print(f"== TOTAL train {len(train_recs)} windows -> {args.out_dir}/train.jsonl ; dev {len(dev)} (TAB) ==")


def _selfcheck():
    """Round-trip on TAB (the char-offset source): recon surface == gold; no span lost to windowing."""
    docs = json.load(open("corpora/tab/echr_dev.json"))
    sp = splitter()
    checked = mism = lost = 0
    for doc in docs[:20]:
        text = doc["text"]
        spans = [(s, e, TYPE2PHRASE[t]) for s, e, t in tab_gold(doc)]
        words = list(sp(text))
        wspans, _, _, bad = _word_spans(words, spans)
        for cs, ce, _ in spans:
            wi = [i for i, (_, s, e) in enumerate(words) if s < ce and e > cs]
            if not wi:
                continue
            recon = " ".join(words[i][0] for i in range(wi[0], wi[-1] + 1))
            if "".join(recon.split()) != "".join(text[cs:ce].split()):
                mism += 1
            checked += 1
        if not bad:   # completeness only where nothing is intentionally dropped (dropped-long spans)
            emitted = sum(len(r["ner"]) for r in doc_to_records(text, spans))
            lost += len(wspans) - emitted
    print(f"selfcheck: {checked} spans, {mism} recon mismatches, {lost} lost "
          f"({'OK' if mism == 0 and lost == 0 else 'FAIL'})")
    assert mism == 0 and lost == 0


def _parse_mix(s):
    """'nemotron=6000,pilener=3000,wikibio=corpora/wikipedia_bio' -> {name: int|path}"""
    out = {}
    for part in s.split(","):
        k, v = part.split("=", 1)
        out[k.strip()] = int(v) if v.strip().isdigit() else v.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="corpora/tab/echr_train.json")
    ap.add_argument("--dev", default="corpora/tab/echr_dev.json")
    ap.add_argument("--out-dir", default="data/pii_span_dataset")
    ap.add_argument("--mix", type=_parse_mix, default=None,
                    help="e.g. nemotron=6000,pilener=3000,wikibio=corpora/wikipedia_bio")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
