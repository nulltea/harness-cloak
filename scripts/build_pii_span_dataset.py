"""Phase 0 — build the PII span-detection training set in gliner format.

Default (no --mix): TAB-only (single-domain fine-tune v1 input).
With --mix: multi-domain mix (fine-tune v2) — TAB (anchor) + auxiliary sources mapped onto the 8 TAB
types, plus an optional diverse-label slice (Pile-NER, own labels) for open-label generality. TAB stays
dominant by a token/window cap. See research-wiki/training/2026-07-04-ft-detector-quasi.md.

NOTE on the `--mix nemotron=N,pilener=N` numbers: N is only the HF **stream depth** (`.take(N)`), NOT the
kept-window count. Kept windows for a mapped aux source are capped by MIX_RATIO relative to TAB
(cap = MIX_RATIO[src]/MIX_RATIO["tab"] * n_tab); raising N above what fills that cap adds nothing. Set N
generously — large enough to fill the cap after empties/offset-drops.

With --balance-rare (fine-tune v3, generality-first): (1) after the mapped schema pool (TAB + Nemotron +
wikibio) is assembled, upsample-with-replacement every window containing a scarce gap type
(MISC/DEM/QUANTITY), bounded to a global <=x2 duplication ceiling (a bounded nudge, not full balance —
DATETIME:MISC is ~10:1, which <=x2 can't equalize without memorizing scarce MISC); (2) size the Pile-NER
slice to PILE_FRAC of the FINAL total, POST-balance, so per-type upsampling (TAB-8 windows only) never
dilutes the diverse-label generality signal. Under --balance-rare, `pilener=N` sets only the stream depth;
set it high enough to fill PILE_FRAC. See research-wiki/training/2026-07-04-ft-detector-large-balanced.md.

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

# v3 (--balance-rare): Pile-NER slice as a fraction of the FINAL (post-balance) total. 0.25 = generality-first
# (v2's proven generality lever, raised from ~10%). Only used under --balance-rare; else pilener uses MIX_RATIO.
PILE_FRAC = 0.25
# scarce TAB-8 gap types upsampled by --balance-rare (v2's lowest-recall QUASI types).
RARE_TYPES = {"MISC", "DEM", "QUANTITY"}

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


def _finalize(tag, recs, conflicts, dropped, tok):
    """Drop windows exceeding the subword budget (rare — long single tokens in structured docs),
    print stats, return the kept windows. Guarantees the output has 0 over-budget windows."""
    kept, lens, over = [], [], 0
    for w in recs:
        n = len(tok(" ".join(w["tokenized_text"]), add_special_tokens=False)["input_ids"])
        if n > MAX_SUBWORD:
            over += 1
        else:
            kept.append(w)
            lens.append(n)
    lens.sort()
    p99 = lens[int(0.99 * (len(lens) - 1))] if lens else 0
    types = Counter(p.split(",")[0][:14] for r in kept for _, _, p in r["ner"])
    print(f"  [{tag}] {len(kept)} windows, {sum(len(r['ner']) for r in kept)} spans | "
          f"subword p99 {p99}, max {max(lens or [0])} | over-budget dropped: {over} | "
          f"conflicts {conflicts}, dropped_long {dropped}")
    print(f"       labels: {dict(types.most_common(8))}")
    return kept


def _balance_rare(recs, rng):
    """Upsample-with-replacement schema-pool windows containing a scarce gap type (RARE_TYPES), one extra
    copy each, under a GLOBAL <=x2 duplication ceiling: a window already present twice (e.g. from the
    wikibio oversample) gets no further copy, so nothing ever reaches x3. A bounded nudge (not full
    balance) — stays inside the memorization-safe gap the v2 memorize probe established (MISC train 0.926
    vs test 0.895 at <=x2)."""
    rare_phrases = {p for p, t in GLINER_LABELS.items() if t in RARE_TYPES}
    key = lambda r: json.dumps(r, sort_keys=True)
    counts = Counter(key(r) for r in recs)          # pre-balance multiplicity (wikibio dups already here)
    out, seen, added = list(recs), set(), 0
    for r in recs:
        if not any(sp[2] in rare_phrases for sp in r["ner"]):
            continue
        k = key(r)
        if counts[k] >= 2 or k in seen:             # already at the <=x2 ceiling (or copied this pass)
            continue
        out.append(r)
        seen.add(k)
        added += 1
    rng.shuffle(out)
    print(f"  [balance-rare] +{added} copies of MISC/DEM/QUANTITY windows (global <=x2); "
          f"pool {len(recs)} -> {len(out)}")
    return out


def _log_shares(recs, out_dir):
    """Realized per-type shares (the ORG-dilution diagnostic, spec v5). A window is 'TAB-schema' if it
    carries any of the 8 TAB label phrases; Pile windows carry diverse labels (not in GLINER_LABELS).
    Reports, per TAB-8 type: window_count (windows containing ≥1 span of that type), mention_count,
    token_count (summed span word-length); plus TAB-schema window share and per-type window share of the
    FINAL total. Writes build_shares.json + prints — this is what ties any ORG change to the share it
    was meant to control (Codex R1/R2)."""
    tab_phrases = set(GLINER_LABELS)                       # the 8 TAB label-phrase strings
    total = len(recs)
    win, men, tok_ct = Counter(), Counter(), Counter()
    tab_windows = 0
    for r in recs:
        types = set()
        for s, e, ph in r["ner"]:
            t = GLINER_LABELS.get(ph)                      # None for Pile's diverse (non-TAB) labels
            if t is None:
                continue
            men[t] += 1
            tok_ct[t] += e - s + 1
            types.add(t)
        for t in types:
            win[t] += 1
        if types:
            tab_windows += 1
    shares = {
        "total_windows": total,
        "tab_schema_windows": tab_windows,
        "tab_schema_window_share": round(tab_windows / max(total, 1), 4),
        "per_type": {t: {"windows": win[t], "window_share": round(win[t] / max(total, 1), 4),
                         "mentions": men[t], "tokens": tok_ct[t]}
                     for t in sorted(win)},
    }
    json.dump(shares, open(os.path.join(out_dir, "build_shares.json"), "w"), indent=2)
    org = shares["per_type"].get("ORG", {})
    print(f"  [shares] total={total} tab-schema-window-share={shares['tab_schema_window_share']:.3f} "
          f"| ORG windows={org.get('windows')} share={org.get('window_share')}")
    print(f"       per-type window_share: "
          f"{ {t: shares['per_type'][t]['window_share'] for t in shares['per_type']} }")
    return shares


def build(args):
    rng = random.Random(SEED)
    tok = _tokenizer()
    pile_frac = args.pile_frac if args.pile_frac is not None else PILE_FRAC

    print(f"== source: tab ({args.train}) ==")
    tab, c, d = source_windows(tab_source(args.train))
    train_recs = _finalize("tab", tab, c, d, tok)
    n_tab = len(train_recs)

    if args.mix:
        pile_spec = None
        for name, spec in args.mix.items():
            if name == "pilener" and args.balance_rare:
                pile_spec = spec                      # sized AFTER balancing (post-balance fraction)
                continue
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
                train_recs += _finalize(name, recs, c, d, tok)

        # v3: per-type balancing on the schema pool (TAB-8 windows only), BEFORE the diverse Pile slice
        if args.balance_rare:
            train_recs = _balance_rare(train_recs, rng)

        # v3/v5: Pile-NER sized to pile_frac of the FINAL total, POST-balance (own labels; generality lever)
        if pile_spec is not None:
            n_pile = int(round(pile_frac / (1 - pile_frac) * len(train_recs)))
            print(f"== source: pilener (post-balance target ~{n_pile} ~= {pile_frac:.0%} of final) ==")
            recs, c, d = source_windows(pilener_source(pile_spec), cap=n_pile * 3)
            recs = [r for r in recs if r["ner"]]
            if len(recs) > n_pile:
                recs = rng.sample(recs, n_pile)
            if recs:
                train_recs += _finalize("pilener", recs, c, d, tok)

    rng.shuffle(train_recs)
    os.makedirs(args.out_dir, exist_ok=True)
    shares = _log_shares(train_recs, args.out_dir)        # v5: realized per-type shares (ORG-dilution diagnostic)
    if args.min_tab_share > 0 and shares["tab_schema_window_share"] < args.min_tab_share:
        # ponytail: fail-fast, operator lowers --pile-frac (the sweep varies it) — no auto-rebalance machinery.
        raise SystemExit(f"TAB-schema window share {shares['tab_schema_window_share']:.3f} < "
                         f"--min-tab-share {args.min_tab_share}: lower --pile-frac (Pile dilutes TAB/ORG "
                         f"share — the ORG-regression lever, spec v5). Shares in {args.out_dir}/build_shares.json")
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
    _balance_selfcheck()
    _shares_selfcheck()


def _shares_selfcheck():
    """_log_shares: TAB-schema windows counted, Pile (diverse-label) windows excluded from TAB share,
    per-type ORG share correct."""
    import tempfile
    org, misc, person = TYPE2PHRASE["ORG"], TYPE2PHRASE["MISC"], TYPE2PHRASE["PERSON"]
    recs = [
        {"tokenized_text": ["a"], "ner": [[0, 0, org], [1, 2, person]]},   # TAB window w/ ORG (2-word person)
        {"tokenized_text": ["b"], "ner": [[0, 0, misc]]},                  # TAB window, no ORG
        {"tokenized_text": ["c"], "ner": [[0, 0, "animal"]]},              # Pile diverse label -> not TAB
    ]
    d = tempfile.mkdtemp()
    s = _log_shares(recs, d)
    assert s["total_windows"] == 3
    assert s["tab_schema_windows"] == 2 and abs(s["tab_schema_window_share"] - 2/3) < 1e-3   # share rounded to 4dp
    assert s["per_type"]["ORG"]["windows"] == 1 and abs(s["per_type"]["ORG"]["window_share"] - 1/3) < 1e-3
    assert s["per_type"]["PERSON"]["tokens"] == 2          # span [1,2] -> 2 words
    assert "animal" not in s["per_type"]                   # diverse labels are not TAB-8 types
    print("shares selfcheck: OK (TAB windows counted, Pile excluded, ORG share + token-len correct)")


def _balance_selfcheck():
    """--balance-rare invariants: rare-type windows get one copy (->x2), common-only untouched, and the
    global <=x2 ceiling holds for windows already doubled (wikibio-style)."""
    rng = random.Random(0)
    misc, dem, person = TYPE2PHRASE["MISC"], TYPE2PHRASE["DEM"], TYPE2PHRASE["PERSON"]
    common = {"tokenized_text": ["a"], "ner": [[0, 0, person]]}   # common only -> not upsampled
    rare1 = {"tokenized_text": ["b"], "ner": [[0, 0, misc]]}      # rare, single -> +1 copy
    rare2 = {"tokenized_text": ["c"], "ner": [[0, 0, dem]]}       # rare, single -> +1 copy
    dup = {"tokenized_text": ["d"], "ner": [[0, 0, misc]]}        # rare but already x2 -> NOT copied again
    out = _balance_rare([common, rare1, rare2, dup, dict(dup)], rng)
    cnt = Counter(json.dumps(r, sort_keys=True) for r in out)
    k = lambda r: json.dumps(r, sort_keys=True)
    assert max(cnt.values()) <= 2, f"global <=x2 violated: {cnt}"
    assert cnt[k(rare1)] == 2 and cnt[k(rare2)] == 2, "rare single not doubled"
    assert cnt[k(common)] == 1, "common-only window was upsampled"
    assert cnt[k(dup)] == 2, "already-x2 window pushed past x2"
    print("balance selfcheck: OK (rare doubled, common untouched, global <=x2)")


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
    ap.add_argument("--balance-rare", action="store_true",
                    help="v3: upsample MISC/DEM/QUANTITY windows (<=x2) + size Pile-NER to --pile-frac "
                         "post-balance (generality-first). Off = v2 behaviour.")
    ap.add_argument("--pile-frac", type=float, default=None,
                    help=f"v5: Pile-NER fraction of the final set under --balance-rare (default {PILE_FRAC}). "
                         "Sweep this: 0.10 (~v2), 0.15/0.18/0.22, 0.25 (~v4).")
    ap.add_argument("--min-tab-share", type=float, default=0.0,
                    help="v5: if >0, abort the build when the realized TAB-schema window share falls below "
                         "this (the ORG-dilution guard). Off (0) = log shares only, don't enforce.")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
