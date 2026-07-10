import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))


def test_detect_docs_dedups_lattice_spans_by_entry(monkeypatch):
    import build_probes
    from cloak.profile_match import MatchResult

    class FakeGliner:
        def predict_entities(self, piece, labels, threshold):
            return [
                {"text": "blorbitis", "label": "condition"},
                {"text": "blorb inflammation", "label": "condition"},
                {"text": "glimmerosis", "label": "condition"},
                {"text": "ghostitis", "label": "condition"},      # matcher abstains
                {"text": "Ann", "label": "name"},
                {"text": "Ann", "label": "name"},                  # surface-dup placeholder
            ]

    monkeypatch.setattr(build_probes, "GLINER_LOADER",
                        lambda model: FakeGliner(), raising=False)
    # if _detect_docs imports GLiNER inline, monkeypatch the gliner module instead:
    fake_mod = types.SimpleNamespace(GLiNER=types.SimpleNamespace(
        from_pretrained=lambda model: FakeGliner()))
    monkeypatch.setitem(sys.modules, "gliner", fake_mod)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)))

    def fake_match(items, **kwargs):
        entry_of = {"blorbitis": "blorbitis", "blorb inflammation": "blorbitis",
                    "glimmerosis": "glimmerosis"}
        out = {}
        for surface, rtype, ctx in items:
            from cloak.profile_match import span_key
            e = entry_of.get(surface.lower())
            out[span_key(surface, rtype)] = (
                MatchResult(["organ disease"], "exact", True, 1.0, e) if e else None)
        return out

    monkeypatch.setattr(build_probes, "match_spans_batch", fake_match, raising=False)

    docs = [{"id": "d1", "text": "Ann has blorbitis, blorb inflammation, glimmerosis, "
                                 "ghostitis. Ann rests."}]
    got = build_probes._detect_docs(docs, "any-model", 0.3)
    spans = got["d1"]
    lattice = [s for s in spans if s["role"] == "lattice"]
    # blorbitis + blorb inflammation collapse to one span; ghostitis dropped (abstain)
    assert [s["surface"] for s in lattice] == ["blorbitis", "glimmerosis"]
    assert lattice[0]["entry"] == "blorbitis"
    # placeholder spans still deduped by surface, kept without entry
    names = [s for s in spans if s["type"] == "PERSON"]
    assert len(names) == 1 and "entry" not in names[0]
