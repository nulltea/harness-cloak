import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from build_probes import split_by_fact, validate_probes  # noqa: E402

from cloak.train.reward import canon  # noqa: E402


def test_validate_probes_keep_and_reject():
    cands = [{"surface": "metformin 500mg", "question": "What dose?"},   # keep
             {"surface": "March 3", "question": "When?"},                # ceiling reject
             {"surface": "chest pain", "question": "What symptom?"}]     # floor reject
    hi = [1.0, 0.2, 1.0]   # f1 vs ceiling anchor out_final(doc_orig)
    lo = [0.0, 0.0, 0.9]   # f1 vs floor anchor out_final(all_placeholder)
    kept, rej_c, rej_f = validate_probes(cands, hi, lo, th=0.5)
    assert [p["surface"] for p in kept] == ["metformin 500mg"]
    assert [p["surface"] for p in rej_c] == ["March 3"]
    assert [p["surface"] for p in rej_f] == ["chest pain"]


def test_split_keeps_all_questions_of_a_fact_on_one_side():
    # 4 facts, 2 questions each; a fact's questions must never straddle the split
    kept = []
    for s in ["Oslo", "Dr. Kumar", "42 mg", "March 3"]:
        kept += [{"surface": s, "question": "q1"}, {"surface": s, "question": "q2"}]
    train, heldout, n_train_facts = split_by_fact(kept, seed=0)
    train_facts = {canon(p["surface"]) for p in train}
    held_facts = {canon(p["surface"]) for p in heldout}
    assert not (train_facts & held_facts)          # no fact spans both splits
    assert n_train_facts == len(train_facts) == 3  # 4 facts, hold out max(1, 4//4)=1
    for p in kept:                                  # every question landed with its fact
        side = train if canon(p["surface"]) in train_facts else heldout
        assert p in side


def test_four_questions_one_fact_is_excluded_under_three_facts():
    # 4 kept questions but ONE fact (canon collapses mg/milligrams) -> < 3 train facts
    kept = [{"surface": s, "question": f"q{i}"} for i, s in
            enumerate(["40 mg", "40 milligrams", "40 mg", "40 milligrams"])]
    _, _, n_train_facts = split_by_fact(kept, seed=0)
    assert n_train_facts == 1 and n_train_facts < 3
