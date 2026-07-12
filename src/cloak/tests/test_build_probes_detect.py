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
    # blorbitis + blorb inflammation collapse to one span; ghostitis abstains -> NOT a
    # probe span, but demoted to placeholder role so the floor still hides it (deleting
    # it leaked the surface to the remote — the synthroid floor-leak bug)
    assert [s["surface"] for s in lattice] == ["blorbitis", "glimmerosis"]
    assert lattice[0]["entry"] == "blorbitis"
    ghost = [s for s in spans if s["surface"] == "ghostitis"]
    assert len(ghost) == 1 and ghost[0]["role"] == "placeholder" and "entry" not in ghost[0]
    # placeholder spans still deduped by surface, kept without entry
    names = [s for s in spans if s["type"] == "PERSON"]
    assert len(names) == 1 and "entry" not in names[0]


def test_detect_docs_demotes_screened_spans_to_placeholder(monkeypatch):
    # a lattice fact whose containing sentence is interrogative (dialogue screening) is
    # excluded from probes but MUST stay in the span list as placeholder role: the floor
    # anonymization hides only listed spans (measured leak: synthroid in aci/D2N002,
    # first mention "how are you doing with the synthroid?")
    import build_probes
    from cloak.profile_match import MatchResult, span_key

    class FakeGliner:
        def predict_entities(self, piece, labels, threshold):
            return [
                {"text": "zoxatol", "label": "drug"},
                {"text": "blorbitis", "label": "condition"},
            ]

    fake_mod = types.SimpleNamespace(GLiNER=types.SimpleNamespace(
        from_pretrained=lambda model: FakeGliner()))
    monkeypatch.setitem(sys.modules, "gliner", fake_mod)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(
        build_probes, "match_spans_batch",
        lambda items, **kw: {span_key(s, t): MatchResult(["organ disease"], "exact",
                                                         True, 1.0, s.lower())
                             for s, t, _ in items},
        raising=False)

    docs = [{"id": "d1", "text": "Are you still taking the zoxatol? Blorbitis is stable."}]
    spans = build_probes._detect_docs(docs, "any-model", 0.3)["d1"]

    zox = [s for s in spans if s["surface"] == "zoxatol"]
    assert len(zox) == 1 and zox[0]["role"] == "placeholder"  # demoted, not deleted
    assert [s["surface"] for s in spans if s["role"] == "lattice"] == ["blorbitis"]
