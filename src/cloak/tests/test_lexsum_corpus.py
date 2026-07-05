"""lexsum loader shape — offline 2-doc fixture (no HF download)."""
import json

from cloak import corpora
from cloak.tasks import TASK_TEMPLATE


def test_lexsum_loader_shape(tmp_path, monkeypatch):
    rows = [
        {"id": "lexsum/0", "corpus": "lexsum",
         "text": "Long summary naming Jane Doe v. Acme Corp in the District Court.",
         "gold_ref": "Jane Doe sued Acme Corp; court ruled for the plaintiff."},
        {"id": "lexsum/1", "corpus": "lexsum",
         "text": "Long summary about the Southern District and the ACLU.",
         "gold_ref": "The ACLU prevailed in the Southern District."},
    ]
    (tmp_path / "lexsum").mkdir()
    with open(tmp_path / corpora.FILES["lexsum"][0], "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    monkeypatch.setattr(corpora, "CORPORA", tmp_path)

    docs = corpora.load_task_docs("lexsum")
    assert len(docs) == 2
    assert docs[0]["text"] and docs[0]["corpus"] == "lexsum"
    assert corpora.refs_of(docs[0]) == [rows[0]["gold_ref"]]
    assert corpora.load_task_docs("lexsum", 1) == rows[:1]


def test_lexsum_task_registered():
    assert "lexsum" in TASK_TEMPLATE
    assert "{doc}" in TASK_TEMPLATE["lexsum"]
