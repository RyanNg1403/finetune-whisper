# src/dataset.py
import json
import glob
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import WhisperProcessor
from src import config
from src.audio_io import load_wav_16k_mono
from src.augment import build_train_augmenter, apply


class WhisperASRDataset(Dataset):
    """Loads (audio, text) from per-split manifests. Optionally augments the waveform on the
    fly (train only) before feature extraction."""

    def __init__(self, dirs, processor: WhisperProcessor, augment=False):
        self.processor = processor
        self.augment = augment
        self.aug = build_train_augmenter() if augment else None
        self.rows = []
        for d in dirs:
            for m in glob.glob(str(d / "manifest*.jsonl")):
                for line in open(m):
                    r = json.loads(line)
                    r["_dir"] = str(d)
                    self.rows.append(r)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        wav = load_wav_16k_mono(f"{r['_dir']}/{r['audio']}")
        if self.augment:
            wav = apply(self.aug, wav)
        feats = self.processor.feature_extractor(
            wav, sampling_rate=config.SAMPLE_RATE).input_features[0]
        labels = self.processor.tokenizer(r["text"]).input_ids
        return {"input_features": feats, "labels": labels}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features):
        input_feats = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")
        label_feats = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_feats, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # drop the BOS the tokenizer prepends; the model adds it during training
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
