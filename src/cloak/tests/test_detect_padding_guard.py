"""Regression: gliner span->text mapping must drop padding-region phantom spans, not IndexError.

Root cause: the deberta-v3-large PII fine-tune at threshold < ~0.1 emits low-confidence MISC spans
whose token indices exceed the real sequence (observed start=225 into a 203-entry map). Upstream
gliner indexes the map unguarded -> IndexError. See cloak.detect._guarded_map_entities_to_original.
Model-free: exercises the guard directly with fake spans (no 1.7 GB checkpoint load).
"""
from types import SimpleNamespace

from cloak.detect import _guarded_map_entities_to_original


def _span(start, end, score=0.05, label="MISC"):
    return SimpleNamespace(start=start, end=end, score=score, entity_type=label, class_probs=None)


def test_drops_out_of_range_keeps_valid():
    smap = list(range(0, 10))          # 10 tokens -> char idx 0..9
    emap = list(range(1, 11))
    text = "abcdefghij"
    output = [_span(2, 3), _span(10, 10), _span(9, 12), _span(0, 0)]   # 2 valid, 2 overflow
    out = _guarded_map_entities_to_original(
        self=None, outputs=[output], valid_to_orig_idx=[0],
        all_start_token_idx_to_text_idx=[smap], all_end_token_idx_to_text_idx=[emap],
        valid_texts=[text], num_original_texts=1)
    kept = out[0]
    assert len(kept) == 2, f"expected 2 in-range spans, got {len(kept)}: {kept}"
    assert {(e["start"], e["end"]) for e in kept} == {(2, 4), (0, 1)}   # via smap/emap lookups
    # the padding-region spans (start=10 == len, end=12 > len) were dropped, no IndexError


def test_empty_output_and_orig_index_mapping():
    out = _guarded_map_entities_to_original(
        self=None, outputs=[[], [_span(0, 0)]], valid_to_orig_idx=[0, 2],
        all_start_token_idx_to_text_idx=[[5], [5]], all_end_token_idx_to_text_idx=[[6], [6]],
        valid_texts=["x", "y"], num_original_texts=3)
    assert out[0] == [] and out[1] == [] and len(out[2]) == 1   # valid_i=1 -> orig index 2


if __name__ == "__main__":
    test_drops_out_of_range_keeps_valid()
    test_empty_output_and_orig_index_mapping()
    print("padding-guard tests: OK")
