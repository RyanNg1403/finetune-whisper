# src/train.py
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse
import numpy as np
import torch
from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)
from src import config
from src.dataset import WhisperASRDataset, DataCollatorSpeechSeq2SeqWithPadding
from src.metrics import compute_wer, term_recall
from src.normalize import load_terms


def build_compute_metrics(processor):
    ts = load_terms()

    def _fn(pred):
        ids = pred.predictions
        label_ids = np.where(pred.label_ids == -100, processor.tokenizer.pad_token_id, pred.label_ids)
        hyps = processor.batch_decode(ids, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": compute_wer(refs, hyps),
                "term_recall": term_recall(refs, hyps, ts)["overall"]}
    return _fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    proc = WhisperProcessor.from_pretrained(config.MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(config.MODEL_ID)
    # base.en is English-only; leave generation_config as-is (setting language/task raises).

    train_ds = WhisperASRDataset([config.TRAIN_DIR], proc, augment=True)
    eval_ds = WhisperASRDataset([config.VAL_CLEAN_DIR], proc, augment=False)
    collator = DataCollatorSpeechSeq2SeqWithPadding(proc)

    targs = Seq2SeqTrainingArguments(
        output_dir=str(config.CHECKPOINTS),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=200,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=None,          # keep EVERY epoch checkpoint (user decision)
        predict_with_generate=True,
        generation_max_length=128,
        fp16=False, bf16=False,         # fp32 on MPS
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=25,
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator, compute_metrics=build_compute_metrics(proc),
        processing_class=proc.feature_extractor,
    )
    trainer.train()
    trainer.save_model(str(config.CHECKPOINTS / "final"))
    proc.save_pretrained(str(config.CHECKPOINTS / "final"))
    print("done; checkpoints in", config.CHECKPOINTS)


if __name__ == "__main__":
    main()
