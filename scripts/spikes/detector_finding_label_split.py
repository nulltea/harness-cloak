"""Spike: does adding finding/sign GLiNER labels pull exam-findings off the
single "condition" label, while real diagnoses stay "condition"?

If yes, the detector label-schema fix is viable: map the finding labels to a
non-controlled type so edema/erythema/immunocompromised/acute exacerbation
never become relation-eligible controlled conditions.

Read-only w.r.t. repo files: constructs a Detector, then queries GLiNER with an
extended label set directly. One model load, GPU.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from cloak.detect import Detector, QA_V2_CLINICAL_LABELS, _chunks
from cloak.corpora import load_task_docs

BASE = list(QA_V2_CLINICAL_LABELS)              # includes "condition"
FINDING_LABELS = ["symptom", "physical examination finding", "clinical sign"]
LABELS = BASE + FINDING_LABELS

TARGETS = {"edema", "erythema", "immunocompromised", "acute exacerbation",
           "arthritis", "hypothyroidism", "kidney transplant", "hyperthyroidism"}


def main():
    text = next(d["text"] for d in load_task_docs("clinical", 3) if d["id"] == "aci/D2N002")
    det = Detector(gliner_model="knowledgator/gliner-pii-large-v1.0",
                   threshold=0.35, profile="clinical", label2type=QA_V2_CLINICAL_LABELS)

    # raw per-label scores (no overlap dedup) across chunks
    chunks = [c for _off, c in _chunks(text, max_words=det.max_words)]
    outputs = det.gliner.batch_predict_entities(chunks, LABELS, threshold=0.30,
                                                batch_size=det.batch_size)
    by_surface = {}
    for entities in outputs:
        for e in entities:
            s = e["text"].strip().lower()
            if s in TARGETS:
                by_surface.setdefault(s, []).append((e["label"], round(e["score"], 3)))

    print(f"labels queried: {LABELS}\n")
    for surf in ["arthritis", "hypothyroidism", "kidney transplant", "hyperthyroidism",
                 "edema", "erythema", "immunocompromised", "acute exacerbation"]:
        rows = sorted(set(by_surface.get(surf, [])), key=lambda r: -r[1])
        winner = rows[0][0] if rows else "(none)"
        tag = "FINDING" if winner in FINDING_LABELS else ("condition" if winner == "condition" else winner)
        print(f"{surf:22} winner={tag:10} scores={rows}")


if __name__ == "__main__":
    main()
