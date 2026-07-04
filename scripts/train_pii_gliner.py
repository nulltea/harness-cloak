"""Phase 2 · Arm B — fine-tune a GLiNER checkpoint on the TAB PII span dataset (Phase 0 output).

Default init = knowledgator/gliner-pii-base-v1.0 (best off-the-shelf on P1 QUASI and P7 generality;
see docs/research/learned-PII-detection.md §5.1b/§5.1c). Keeps the open-label interface, so the
fine-tuned model still takes user-defined type phrases at inference (tailorability / P6).

Windows and label phrases come from scripts/build_pii_span_dataset.py (run that first).
Per-epoch checkpoints are saved; final selection is the dev gate (QUASI any-recall at precision
>= 0.70), run per checkpoint via:
    PYTHONPATH=src .venv/bin/python scripts/latticecloak_detection_gate.py \
        --corpus corpora/tab/echr_dev.json --gliner-model data/models/pii_gliner/checkpoint-XXXX

Full run:  PYTHONPATH=src .venv/bin/python -u scripts/train_pii_gliner.py
Smoke:     ... scripts/train_pii_gliner.py --limit 32 --max-steps 2 --epochs 1 --out /tmp/pii_gliner_smoke
"""
import argparse
import json
import os


def load_jsonl(path, limit=0):
    data = [json.loads(l) for l in open(path)]
    return data[:limit] if limit else data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="knowledgator/gliner-pii-base-v1.0")
    ap.add_argument("--data-dir", default="data/pii_span_dataset")
    ap.add_argument("--out", default="data/models/pii_gliner")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)       # effective batch = batch-size * this
    ap.add_argument("--lr", type=float, default=1e-5)          # transformer backbone
    ap.add_argument("--others-lr", type=float, default=5e-5)   # span/projection layers
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")           # resume from latest checkpoint in --out
    ap.add_argument("--resume-from-checkpoint", default=None)   # or resume from an explicit path
    ap.add_argument("--overwrite", action="store_true")        # allow writing into a non-empty --out
    ap.add_argument("--max-steps", type=int, default=-1)       # >0 overrides epochs (smoke)
    ap.add_argument("--limit", type=int, default=0)            # cap train examples (smoke)
    args = ap.parse_args()
    smoke = args.max_steps > 0

    # output-dir hygiene: refuse to mix artifacts unless resuming or explicitly overwriting
    if not smoke and os.path.isdir(args.out) and os.listdir(args.out) \
            and not (args.resume or args.overwrite):
        raise SystemExit(f"{args.out} is non-empty; pass --resume or --overwrite (guards against "
                         "mixing checkpoints from different runs)")

    import torch
    from gliner import GLiNER
    from gliner.data_processing.collator import SpanDataCollator
    from gliner.training import Trainer, TrainingArguments
    from transformers import set_seed
    set_seed(args.seed)

    train = load_jsonl(os.path.join(args.data_dir, "train.jsonl"), args.limit)
    dev = load_jsonl(os.path.join(args.data_dir, "dev.jsonl"))
    print(f"train={len(train)} dev={len(dev)} windows | init={args.init} | seed={args.seed}",
          flush=True)

    model = GLiNER.from_pretrained(args.init)
    if torch.cuda.is_available():
        model = model.to("cuda")
    collator = SpanDataCollator(model.config, data_processor=model.data_processor,
                                prepare_labels=True)

    # train-time preflight: the dataset's subword budget was checked against deberta-v3-base; verify
    # it still holds under THIS --init's actual tokenizer (guards a mismatched init).
    tok = model.data_processor.transformer_tokenizer
    over = [r for r in train + dev
            if len(tok(" ".join(r["tokenized_text"]), add_special_tokens=False)["input_ids"]) > 480]
    if over:
        raise SystemExit(f"{len(over)} windows exceed 480 subwords under {args.init}'s tokenizer — "
                         "rebuild the dataset with a smaller WINDOW_WORDS")

    # bf16 only where actually supported; else fp16 on CUDA; else fp32
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    targs = TrainingArguments(
        output_dir=args.out, seed=args.seed,
        learning_rate=args.lr, others_lr=args.others_lr,
        weight_decay=0.01, others_weight_decay=0.01,
        lr_scheduler_type="cosine", warmup_ratio=0.1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        eval_strategy="no" if smoke else "epoch",
        save_strategy="no" if smoke else "epoch", save_total_limit=6,
        bf16=use_bf16, fp16=use_fp16, dataloader_num_workers=0,
        logging_steps=50, report_to="none",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train,
                      eval_dataset=None if smoke else dev, data_collator=collator,
                      processing_class=model.data_processor.transformer_tokenizer)

    if not smoke:  # run manifest for reproducibility / audit
        import gliner, transformers
        json.dump({"init": args.init, "seed": args.seed, "epochs": args.epochs,
                   "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                   "lr": args.lr, "others_lr": args.others_lr,
                   "n_train": len(train), "n_dev": len(dev),
                   "bf16": use_bf16, "fp16": use_fp16,
                   "gliner": gliner.__version__, "transformers": transformers.__version__,
                   "torch": torch.__version__},
                  open(os.path.join(args.out, "run_manifest.json"), "w"), indent=2)

    resume = args.resume_from_checkpoint or (args.resume or None)
    trainer.train(resume_from_checkpoint=resume)
    final = os.path.join(args.out, "final")   # NOT the out root: per-epoch checkpoint-* stay separable
    model.save_pretrained(final)
    print(f"saved final epoch -> {final}", flush=True)
    if not smoke:
        print(f"NEXT — {final} is the LAST epoch, NOT the deployment model. Select the best checkpoint "
              f"on the dev gate (QUASI any-recall at precision >= 0.70), then run the P7 retention "
              f"delta on the winner. For each {args.out}/checkpoint-* AND {final}:\n"
              f"  scripts/latticecloak_detection_gate.py --corpus corpora/tab/echr_dev.json "
              f"--gliner-model <ckpt>\n"
              f"  scripts/spikes/pii_zeroshot_generality.py --gliner-model <selected-ckpt>\n"
              f"(dev-loss-best is NOT the selection metric — pick by the gate's QUASI recall.)")


if __name__ == "__main__":
    main()
