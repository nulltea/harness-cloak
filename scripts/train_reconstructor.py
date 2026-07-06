"""Fine-tune flan-t5-base + LoRA on residue-edit triples (scripts/build_reconstructor_data.py).
Follows scripts/train_pii_gliner.py's HF-Trainer+PEFT pattern. Local ROCm GPU, one process.

--split all   : train on every row (cross-domain checkpoint, e.g. reconstructor_v1 on clinical).
--split train : keep only train-split docs (in-domain held-out checkpoints); reads the
                per-corpus data/recon_train_ids_<corpus>.txt the builder writes, so the eval's
                --doc-split heldout complement is genuinely unseen.

Run: PYTHONPATH=src .venv/bin/python -u scripts/train_reconstructor.py \
       --train data/reconstructor_clinical.jsonl --out data/models/reconstructor_v1 \
       --epochs 4 --bs 4 > results/train_reconstructor.log 2>&1
"""
import argparse, json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments)

BASE = "google/flan-t5-base"
PROMPT = ("Restore the original terms in the CLINICAL/LEGAL answer below. Replace each "
          "generalized mention with its original from the RESTORE map; copy everything else "
          "verbatim; if a mapped term is not present, leave the text unchanged.\n\n{input}")


def load(path, tok, split="all", noop_cap=0.3):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if split == "train":
        # In-domain held-out training: keep only train-split docs so the eval's heldout
        # complement is unseen. Read the per-corpus split file(s) the builder wrote.
        corpora = {r["corpus"] for r in rows}
        train_ids = set()
        for c in corpora:
            p = Path(f"data/recon_train_ids_{c}.txt")
            if p.exists():
                train_ids |= set(p.read_text().split())
        before = len(rows)
        rows = [r for r in rows if r["doc_id"] in train_ids]
        print(f"--split train: kept {len(rows)}/{before} rows on train-split docs")
    # Keep all residue-positive edits. For no-ops (Round-1 #5 / Round-2 #5): abstention on
    # high-risk rejects (D-4-like / scalar/date) is a SAFETY signal, not class-balance noise
    # — keep ALL of those uncapped; cap only generic no-ops at noop_cap of the total.
    pos = [r for r in rows if not r.get("is_noop")]
    risky = [r for r in rows if r.get("is_noop") and r.get("high_risk_noop")]
    generic = [r for r in rows if r.get("is_noop") and not r.get("high_risk_noop")]
    keep_generic = generic[:int(noop_cap / (1 - noop_cap) * len(pos))] if pos else generic[:len(generic)//3]
    rows = pos + risky + keep_generic
    print(f"train rows: {len(pos)} positive + {len(risky)} high-risk no-op (kept all) + "
          f"{len(keep_generic)} generic no-op (of {len(generic)})")
    def enc(r):
        x = tok(PROMPT.format(input=r["input"]), truncation=True, max_length=1024)
        y = tok(text_target=r["target"], truncation=True, max_length=1024)
        x["labels"] = y["input_ids"]
        return x
    return Dataset.from_list(rows).map(enc, remove_columns=["input", "target", "corpus",
                                       "doc_id", "n_residue", "n_edits", "is_noop",
                                       "high_risk_noop"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--split", choices=["all", "train"], default="all")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q", "v"], task_type="SEQ_2_SEQ_LM"))
    model.print_trainable_parameters()
    ds = load(args.train, tok, split=args.split)
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(output_dir=args.out, num_train_epochs=args.epochs,
            per_device_train_batch_size=args.bs, learning_rate=2e-4, bf16=True,
            logging_steps=10, save_strategy="epoch", report_to=[]),
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model))
    trainer.train()
    trainer.save_model(args.out); tok.save_pretrained(args.out)
    print(f"saved LoRA reconstructor -> {args.out}")


if __name__ == "__main__":
    main()
